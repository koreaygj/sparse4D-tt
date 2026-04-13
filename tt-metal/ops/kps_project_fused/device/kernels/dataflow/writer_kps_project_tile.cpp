// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Writer kernel for TILE-based kps_project_fused
//
// Reads projected TILE [32, 32] f32 from compute output CB.
// Each tile has projected [NUM_PTS, 4] data (rows 0..12, cols 0..3):
//   col 0 = proj_x, col 1 = proj_y, col 2 = proj_z, col 3 = unused
//
// Performs scalar perspective divide + grid normalization + clamp,
// then writes output [NC, N, 1, NUM_PTS*2] ROW_MAJOR f32.
// =============================================================================

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// Read f32 from TILE-formatted data
inline float tile_get_f32(volatile tt_l1_ptr uint32_t* tile, uint32_t row, uint32_t col) {
    uint32_t face = ((row >= 16) ? 2 : 0) + ((col >= 16) ? 1 : 0);
    uint32_t fr = row & 15;
    uint32_t fc = col & 15;
    uint32_t bits = tile[face * 256 + fr * 16 + fc];
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

void kernel_main() {
    uint32_t out_addr       = get_arg_val<uint32_t>(0);
    uint32_t wh_addr        = get_arg_val<uint32_t>(1);
    uint32_t num_anchors    = get_arg_val<uint32_t>(2);
    uint32_t anchor_offset  = get_arg_val<uint32_t>(3);
    uint32_t total_anchors  = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out      = get_compile_time_arg_val(0);  // projected tile from compute
    constexpr uint32_t out_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t NC          = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_PTS     = get_compile_time_arg_val(3);
    constexpr uint32_t wh_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t cb_wh_scratch = get_compile_time_arg_val(5); // scratch CB for wh data

    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr auto wh_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_size);
    const auto wh_accessor = TensorAccessor(wh_args, wh_addr, wh_page_size);

    constexpr uint32_t wh_page_stride = wh_page_size / sizeof(float);

    // Load image_wh into L1
    uint32_t wh_l1_addr = get_write_ptr(cb_wh_scratch);
    for (uint32_t page = 0; page < NC; page++) {
        uint64_t noc_addr = wh_accessor.get_noc_addr(page);
        noc_async_read(noc_addr, wh_l1_addr + page * wh_page_size, wh_page_size);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr float* wh_data = reinterpret_cast<volatile tt_l1_ptr float*>(wh_l1_addr);

    // Process each anchor × camera projected tile
    for (uint32_t a_idx = 0; a_idx < num_anchors; a_idx++) {
        uint32_t a = anchor_offset + a_idx;

        for (uint32_t c = 0; c < NC; c++) {
            // Wait for projected tile from compute
            cb_wait_front(cb_out, 1);
            volatile tt_l1_ptr uint32_t* proj_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out));

            float inv_w = 1.0f / wh_data[c * wh_page_stride + 0];
            float inv_h = 1.0f / wh_data[c * wh_page_stride + 1];

            uint32_t out_scratch = wh_l1_addr + NC * wh_page_size;
            volatile tt_l1_ptr float* out = reinterpret_cast<volatile tt_l1_ptr float*>(out_scratch);

            for (uint32_t p = 0; p < NUM_PTS; p++) {
                float proj_x = tile_get_f32(proj_tile, p, 0);
                float proj_y = tile_get_f32(proj_tile, p, 1);
                float proj_z = tile_get_f32(proj_tile, p, 2);

                float z_safe = proj_z > 1e-5f ? proj_z : 1e-5f;
                float inv_z = 1.0f / z_safe;
                float ndx = proj_x * inv_z;
                float ndy = proj_y * inv_z;

                float grid_x = ndx * inv_w * 2.0f - 1.0f;
                float grid_y = ndy * inv_h * 2.0f - 1.0f;

                if (grid_x < -2.0f) grid_x = -2.0f;
                if (grid_x >  2.0f) grid_x =  2.0f;
                if (grid_y < -2.0f) grid_y = -2.0f;
                if (grid_y >  2.0f) grid_y =  2.0f;

                out[p * 2 + 0] = grid_x;
                out[p * 2 + 1] = grid_y;
            }

            cb_pop_front(cb_out, 1);

            // Write output page to DRAM
            uint32_t page_id = c * total_anchors + a;
            uint64_t noc_addr = out_accessor.get_noc_addr(page_id);
            noc_async_write(out_scratch, noc_addr, out_page_size);
            noc_async_write_barrier();
        }
    }
}
