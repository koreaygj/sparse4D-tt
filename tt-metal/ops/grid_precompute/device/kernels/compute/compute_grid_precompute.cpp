// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// grid_precompute — turn the compacted Q14 grid into grid_sample's precomputed
// 6-field form (h0, w0, w_nw, w_ne, w_sw, w_se), per FPN level.
//
// Why this op exists: grid_sample's reader spends ~928 of ~1494 cycles per
// point deriving exactly these six values in soft float on a core with no FPU.
// Feeding it the precomputed form measured 2.78x. The math cannot run in that
// reader (the SFPU is not callable from dataflow cores) and must not run
// before compaction (2.3x more rows; the pre-compaction variant priced out at
// +38.5 ms/frame) — so it runs here, on the rows compaction kept.
//
// ENGINE CHOICE IS THE POINT OF THIS FILE. Every VALUE computation runs on
// the SFPU, which is true IEEE fp32; the FPU matmul path truncates fp32
// operands to ~10 effective mantissa bits (measured on the kps tile path as
// 0.13 px of grid error), and floor() sits exactly at the cliff where that
// truncation flips a pixel index by one. The FPU is used only for the
// selector matmuls at the end, where every operand is either 0/1 or a small
// integer or a weight that ships as bf16 anyway — all exact or already
// rounded coarser than the truncation.
//
//   per level, all SFPU:
//     P   = coords * SCALE + BIAS      per-column affine (x cols W-derived,
//                                      y cols H-derived; Q14 scale folded in)
//     F   = floor(P);  R = P - F
//     m0  = [0 <= F <= C-1]            C is the per-column bound tile
//     m1  = [0 <= F+1 <= C-1]
//     FA0 = (1-R) * m0                 boundary masks FACTORISE into the
//     FA1 = R * m1                     bilinear terms, so the four masked
//                                      weights are outer products of factors
//   then FPU routing:
//     out_j = (FA0xSB0_j + FA1xSB1_j) . (FA0xSH0_j + FA1xSH1_j) + F x SI_j
//
// The selectors are 0/1 matrices that route x/y factor columns into the
// interleaved 6-field output — the same trick _softmax_clp uses. SI routes
// the indices into fields 0..1 where the weight product is zero by
// construction, so one accumulating matmul finishes the tile.
// =============================================================================

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/typecast.h"

void kernel_main() {
    const uint32_t num_levels = get_arg_val<uint32_t>(0);   // 4
    const uint32_t out_tiles  = get_arg_val<uint32_t>(1);   // ceil(K*6 / 32) = 3

    constexpr auto cb_coords = tt::CBIndex::c_0;   // f32, reader-built, reused all levels
    constexpr auto cb_const  = tt::CBIndex::c_1;   // f32 stream: per level SCALE, BIAS, C, then per j: SB0 SB1 SH0 SH1 SI
    constexpr auto cb_F      = tt::CBIndex::c_3;
    constexpr auto cb_R      = tt::CBIndex::c_5;
    constexpr auto cb_FA0    = tt::CBIndex::c_6;
    constexpr auto cb_FA1    = tt::CBIndex::c_7;
    constexpr auto cb_outw   = tt::CBIndex::c_16;  // bf16 CB: packer converts the weights in hardware
    constexpr auto cb_outi   = tt::CBIndex::c_17;  // int32 CB: SFPU-typecast indices, writer just moves bytes

    constexpr uint32_t d0 = 0, d1 = 1, d2 = 2, d3 = 3;
    constexpr uint32_t ZERO_F32 = 0u;              // bit pattern of 0.0f
    constexpr uint32_t ONE_F32 = 0x3F800000u;

    // Init against a REGULAR CB, not the unpack-to-dest one, and prime the
    // matmul config once. Without this the first matmul the kernel ever runs
    // (level 0's selector routing) read srcA shifted by one column — h0 came
    // back holding x, w0 held the previous point's value — while every later
    // level was perfect, because their preceding state was a clean matmul.
    binary_op_init_common(cb_F, cb_const, cb_outw);
    mm_init(cb_FA0, cb_const, cb_outw);

    cb_wait_front(cb_coords, 1);
    // The whole constant pack is RESIDENT: 27 tiles pushed once by the reader
    // and never popped, addressed by tile index. Streaming them per level cost
    // 68 serialised DMA barriers a call.
    const uint32_t n_const = 3 * num_levels + 5 * out_tiles;
    cb_wait_front(cb_const, n_const);

    for (uint32_t l = 0; l < num_levels; l++) {
        // ---- P = coords*SCALE + BIAS ; F = floor(P) ; R = P - F -----------
        // One acquire, entirely SFPU: mul, add, a dest copy, floor, sub.
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_coords);
        copy_tile(cb_coords, 0, d0);
        copy_tile_to_dst_init_short(cb_const);
        copy_tile(cb_const, l, d1);                // SCALE_l
        mul_binary_tile_init();
        mul_binary_tile(d0, d1, d0);
        copy_tile_to_dst_init_short(cb_const);
        copy_tile(cb_const, num_levels + l, d1);   // BIAS_l
        add_binary_tile_init();
        add_binary_tile(d0, d1, d0);               // P
        copy_dest_values_init();
        copy_dest_values(d0, d1);                  // d1 = P
        rounding_op_tile_init();
        floor_tile(d1);                            // d1 = F
        sub_binary_tile_init();
        sub_binary_tile(d0, d1, d2);               // d2 = R
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_F);           // packer format is global state;
                                                   // the previous level ended on int32
        cb_reserve_back(cb_F, 1);
        pack_tile(d1, cb_F);
        cb_reserve_back(cb_R, 1);
        pack_tile(d2, cb_R);
        tile_regs_release();
        cb_push_back(cb_F, 1);
        cb_push_back(cb_R, 1);

        // ---- FA0 and FA1 in one acquire -----------------------------------
        // Three rounds used to build these (F1 via its own CB, then each
        // factor separately). Merged: F1 lives only in DEST, the copies are
        // grouped so the unpack/datacopy config is programmed once per batch,
        // and the acquire/pack round-trips drop from three to one. Every
        // *_init here is a hardware reconfigure, not a function call — the
        // interleaved version spent more time reprogramming the SFPU than
        // computing.
        //
        // Slot walk (4 fp32 slots): d0=F->F-C->m_hi->m0->FA0 (held to pack),
        // d1=C->(F1-C)->m1->FA1, d2=F(ge)->F1->ge(F1)->R, d3=R->(1-R)->R.
        cb_wait_front(cb_F, 1);
        cb_wait_front(cb_R, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_F);
        copy_tile(cb_F, 0, d0);
        copy_tile(cb_const, 2 * num_levels + l, d1);   // C_l
        copy_tile(cb_F, 0, d2);
        copy_tile(cb_R, 0, d3);
        sub_binary_tile_init();
        sub_binary_tile(d0, d1, d0);               // F - (C-1)
        unary_le_tile_init();
        unary_le_tile(d0, ZERO_F32);               // F <= C-1
        unary_ge_tile_init();
        unary_ge_tile(d2, ZERO_F32);               // F >= 0
        mul_binary_tile_init();
        mul_binary_tile(d0, d2, d0);               // m0
        binop_with_scalar_tile_init();
        rsub_unary_tile(d3, ONE_F32);              // 1 - R
        mul_binary_tile(d0, d3, d0);               // FA0, held in d0
        copy_tile_to_dst_init_short(cb_F);
        copy_tile(cb_F, 0, d2);
        binop_with_scalar_tile_init();
        add_unary_tile(d2, ONE_F32);               // F1 = F + 1, DEST-only
        sub_binary_tile_init();
        sub_binary_tile(d2, d1, d1);               // F1 - (C-1)  (C no longer needed)
        unary_le_tile_init();
        unary_le_tile(d1, ZERO_F32);
        unary_ge_tile_init();
        unary_ge_tile(d2, ZERO_F32);               // F1 >= 0
        mul_binary_tile_init();
        mul_binary_tile(d1, d2, d1);               // m1
        copy_tile_to_dst_init_short(cb_R);
        copy_tile(cb_R, 0, d2);
        mul_binary_tile_init();
        mul_binary_tile(d1, d2, d1);               // FA1 = R * m1
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_FA0, 1);
        pack_tile(d0, cb_FA0);
        cb_reserve_back(cb_FA1, 1);
        pack_tile(d1, cb_FA1);
        tile_regs_release();
        cb_push_back(cb_FA0, 1);
        cb_push_back(cb_FA1, 1);

        cb_pop_front(cb_R, 1);

        // ---- assemble output tiles ----------------------------------------
        cb_wait_front(cb_FA0, 1);
        cb_wait_front(cb_FA1, 1);
        for (uint32_t j = 0; j < out_tiles; j++) {
            // Selectors live in the resident const CB at 3*NL + 5j + {0..4}:
            // SB0, SB1 accumulate the w-side factor into d0; SH0, SH1 the
            // h-side into d1; SI routes the indices into d2. The SFPU then
            // combines d0 = d0*d1 + d2, duplicates it, and typecasts the copy
            // to int32 — so the writer receives weights already bf16 (packer
            // conversion) and indices already integers, and does no float
            // arithmetic at all. Its scalar conversion loop was the op's
            // critical path at 682 of 948 us.
            const uint32_t sel = 3 * num_levels + 5 * j;
            tile_regs_acquire();
            mm_init(cb_FA0, cb_const, cb_outw);
            matmul_tiles(cb_FA0, cb_const, 0, sel + 0, d0);
            matmul_tiles(cb_FA1, cb_const, 0, sel + 1, d0);
            matmul_tiles(cb_FA0, cb_const, 0, sel + 2, d1);
            matmul_tiles(cb_FA1, cb_const, 0, sel + 3, d1);
            matmul_tiles(cb_F, cb_const, 0, sel + 4, d2);
            mul_binary_tile_init();
            mul_binary_tile(d0, d1, d0);
            add_binary_tile_init();
            add_binary_tile(d0, d2, d0);
            copy_dest_values_init();
            copy_dest_values(d0, d1);
            typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>();
            typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>(d1);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_outw);    // f32 dest -> bf16 in the packer
            cb_reserve_back(cb_outw, 1);
            pack_tile(d0, cb_outw);
            pack_reconfig_data_format(cb_outi);    // int32 passthrough
            cb_reserve_back(cb_outi, 1);
            pack_tile(d1, cb_outi);
            tile_regs_release();
            cb_push_back(cb_outw, 1);
            cb_push_back(cb_outi, 1);
        }
        cb_pop_front(cb_FA0, 1);
        cb_pop_front(cb_FA1, 1);
        cb_pop_front(cb_F, 1);
    }
    cb_pop_front(cb_coords, 1);
    cb_pop_front(cb_const, n_const);
}
