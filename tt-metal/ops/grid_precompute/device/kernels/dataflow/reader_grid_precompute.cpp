// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Reader for grid_precompute.
//
// Two jobs, both cheap by construction:
//
//  1. Build this core's coords tile: its slice of the compacted Q14 grid,
//     sign-extended int16 -> f32, x/y interleaved in columns 0..2K-1 exactly as
//     they sit in the row. The RAW integer goes in — the Q14 scale (1/2^14),
//     the pixel scale and the -0.5 offset arrive as per-column SCALE and BIAS
//     tiles the compute kernel applies on the SFPU — eltwise rather than a
//     matmul on purpose, because the FPU matmul truncates fp32 operands and
//     floor() sits exactly where that truncation flips a pixel index.
//
//  2. Stream the constant tiles (affine D per level, bound C per level, and
//     the level-independent selector matrices) from DRAM in the exact order
//     the compute kernel consumes them. They are built ONCE in python — 23
//     f32 tiles, ~92 KB — because building them here in scalar code measured
//     out at ~0.6 ms/frame when this design was priced, and a DMA of the same
//     tiles is microseconds.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

inline void tile_set_f32(volatile tt_l1_ptr uint32_t* tile, uint32_t row, uint32_t col, float val) {
    uint32_t face = ((row >= 16) ? 2 : 0) + ((col >= 16) ? 1 : 0);
    uint32_t bits;
    __builtin_memcpy(&bits, &val, sizeof(float));
    tile[face * 256 + (row & 15) * 16 + (col & 15)] = bits;
}

void kernel_main() {
    const uint32_t cgrid_addr = get_arg_val<uint32_t>(0);
    const uint32_t const_addr = get_arg_val<uint32_t>(1);
    const uint32_t row_start  = get_arg_val<uint32_t>(2);
    const uint32_t num_rows   = get_arg_val<uint32_t>(3);   // <= 32

    constexpr uint32_t cb_coords      = get_compile_time_arg_val(0);
    constexpr uint32_t cb_const       = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scratch     = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_PTS        = get_compile_time_arg_val(3);   // 13
    constexpr uint32_t cgrid_page     = get_compile_time_arg_val(4);   // aligned row bytes
    constexpr uint32_t NUM_LEVELS     = get_compile_time_arg_val(5);
    constexpr uint32_t OUT_TILES      = get_compile_time_arg_val(6);
    constexpr uint32_t CONST_TILE_BYTES = 32 * 32 * 4;

    constexpr auto cgrid_args = TensorAccessorArgs<7>();
    constexpr auto const_args = TensorAccessorArgs<cgrid_args.next_compile_time_args_offset()>();
    const auto cgrid_acc = TensorAccessor(cgrid_args, cgrid_addr, cgrid_page);
    // A TILE-layout f32 tensor's page IS one 4 KB tile, so the accessor hands
    // back whole constant tiles by index.
    const auto const_acc = TensorAccessor(const_args, const_addr, CONST_TILE_BYTES);

    // ---- coords tile ------------------------------------------------------
    const uint32_t scratch = get_write_ptr(cb_scratch);
    for (uint32_t r = 0; r < num_rows; r++) {
        noc_async_read(cgrid_acc.get_noc_addr(row_start + r), scratch + r * cgrid_page, cgrid_page);
    }

    cb_reserve_back(cb_coords, 1);
    const uint32_t coords_l1 = get_write_ptr(cb_coords);
    volatile tt_l1_ptr uint32_t* ct = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(coords_l1);
    // Zero once: rows past num_rows and columns past 2K must not carry stale
    // values into the matmul.
    for (uint32_t i = 0; i < 1024; i++) {
        ct[i] = 0;
    }
    noc_async_read_barrier();

    for (uint32_t r = 0; r < num_rows; r++) {
        volatile tt_l1_ptr int16_t* rp =
            reinterpret_cast<volatile tt_l1_ptr int16_t*>(scratch + r * cgrid_page);
        for (uint32_t c = 0; c < 2 * NUM_PTS; c++) {
            tile_set_f32(ct, r, c, (float)rp[c]);   // raw Q14 integer; scale lives in D
        }
    }
    cb_push_back(cb_coords, 1);

    // ---- constant pack, resident -----------------------------------------
    // All tiles land in one reserve/push: the reads go to distinct offsets, so
    // one barrier covers every tile. Streaming these per level put 68
    // serialised barriers on this core per call.
    const uint32_t n_const = 3 * NUM_LEVELS + 5 * OUT_TILES;
    cb_reserve_back(cb_const, n_const);
    const uint32_t const_l1 = get_write_ptr(cb_const);
    for (uint32_t i = 0; i < n_const; i++) {
        noc_async_read(const_acc.get_noc_addr(i), const_l1 + i * CONST_TILE_BYTES, CONST_TILE_BYTES);
    }
    noc_async_read_barrier();
    cb_push_back(cb_const, n_const);
}
