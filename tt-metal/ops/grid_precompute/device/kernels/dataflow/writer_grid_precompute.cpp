// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Writer for grid_precompute.
//
// Converts the compute kernel's f32 output tiles into grid_sample's
// precomputed-row format and lands them in the per-level output tensors.
//
// The format is the one contract in this op that is bit-for-bit fixed by a
// consumer: grid_sample's reader takes fields 0..1 (h0, w0) as INT16 BITS
// sitting in a bf16-sized slot, and fields 2..5 as real bf16 weights.
//
// This kernel does NO float arithmetic. Doing the conversions here in soft
// float was the op's critical path — 682 of 948 us — so they moved upstream:
// the packer converts the weights f32->bf16 in hardware on the way into
// cb_outw, and the SFPU typecasts the indices to int32 into cb_outi. What is
// left is interleaving uint16 moves from two tiles into the output stick.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

inline uint32_t tile_off(uint32_t row, uint32_t col) {
    uint32_t face = ((row >= 16) ? 2 : 0) + ((col >= 16) ? 1 : 0);
    return face * 256 + (row & 15) * 16 + (col & 15);
}

void kernel_main() {
    // out addrs for up to 4 levels, then row range
    const uint32_t out_addr0 = get_arg_val<uint32_t>(0);
    const uint32_t out_addr1 = get_arg_val<uint32_t>(1);
    const uint32_t out_addr2 = get_arg_val<uint32_t>(2);
    const uint32_t out_addr3 = get_arg_val<uint32_t>(3);
    const uint32_t row_start = get_arg_val<uint32_t>(4);
    const uint32_t num_rows  = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_outw      = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stage     = get_compile_time_arg_val(1);
    constexpr uint32_t cb_outi      = get_compile_time_arg_val(6);
    constexpr uint32_t NUM_PTS      = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_LEVELS   = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_TILES    = get_compile_time_arg_val(4);
    constexpr uint32_t out_page     = get_compile_time_arg_val(5);   // aligned 6*K*2 bytes

    constexpr uint32_t FIELDS = 6;
    constexpr uint32_t row_vals = NUM_PTS * FIELDS;                  // 78

    constexpr auto out_args = TensorAccessorArgs<7>();
    const uint32_t out_addrs[4] = {out_addr0, out_addr1, out_addr2, out_addr3};

    const uint32_t stage = get_write_ptr(cb_stage);

    for (uint32_t l = 0; l < NUM_LEVELS; l++) {
        const auto out_acc = TensorAccessor(out_args, out_addrs[l], out_page);
        cb_wait_front(cb_outw, OUT_TILES);
        cb_wait_front(cb_outi, OUT_TILES);
        const uint32_t w_l1 = get_read_ptr(cb_outw);
        const uint32_t i_l1 = get_read_ptr(cb_outi);

        for (uint32_t r = 0; r < num_rows; r++) {
            volatile tt_l1_ptr uint16_t* row16 =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + r * out_page);
            // FIELD-MAJOR stick: [h0 x K][w0 x K][weights x 4K]. Each field block
            // is contiguous in its source tile within a 16-column face run, so
            // the copy walks whole runs with an incrementing pointer instead of
            // recomputing face arithmetic per element — the per-element version
            // spent ~40 instructions moving each 2-byte value.
            uint32_t c = 0;
            while (c < 2 * NUM_PTS) {                     // index fields, int32 source
                uint32_t run = ((c & ~15u) + 16 < 2 * NUM_PTS) ? ((c & ~15u) + 16 - c)
                                                               : (2 * NUM_PTS - c);
                volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    i_l1 + (c >> 5) * 4096) + tile_off(r, c & 31);
                for (uint32_t i = 0; i < run; i++) {
                    row16[c + i] = (uint16_t)(src[i] & 0xFFFFu);
                }
                c += run;
            }
            while (c < row_vals) {                        // weight fields, bf16 source
                uint32_t lim = (c & ~15u) + 16;
                uint32_t run = (lim < row_vals) ? (lim - c) : (row_vals - c);
                volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                    w_l1 + (c >> 5) * 2048) + tile_off(r, c & 31);
                // both sides 4-byte aligned when c is even, which every run start is
                volatile tt_l1_ptr uint32_t* s32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
                volatile tt_l1_ptr uint32_t* d32 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&row16[c]);
                const uint32_t pairs = run >> 1;
                for (uint32_t i = 0; i < pairs; i++) {
                    d32[i] = s32[i];
                }
                if (run & 1) {
                    row16[c + run - 1] = src[run - 1];
                }
                c += run;
            }
            noc_async_write(stage + r * out_page, out_acc.get_noc_addr(row_start + r), out_page);
        }
        // One barrier per level, not per row: the stage buffer is per-row slots,
        // so nothing is overwritten while writes are in flight.
        noc_async_write_barrier();
        cb_pop_front(cb_outw, OUT_TILES);
        cb_pop_front(cb_outi, OUT_TILES);
    }
}
