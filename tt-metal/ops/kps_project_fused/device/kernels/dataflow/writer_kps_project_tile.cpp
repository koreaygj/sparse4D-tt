// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Writer kernel for TILE-based kps_project_fused
//
// Reads a TILE [32, 32] f32 from the compute kernel, rows 0..12 being the points:
//   col 0 = depth (already used and divided out), col 1 = x, col 2 = y, in Q14 units
//
// The perspective divide, the normalise and the Q14 scale all happen upstream now — the
// divide on the SFPU, the scale folded into the projection matrix by the reader. That
// chain measured 73% of this path when it ran here in scalar soft float on a core with no
// FPU. What is left is the -1 the normalise leaves, the saturation, and the cast.
//
// Output is [NC, N, 1, NUM_PTS*2] ROW_MAJOR, Q14 fixed point in a UINT16 container — the
// same encoding reader_kps_project.cpp produces on the scalar path, and what grid_sample
// and grid_compact decode. Writing f32 here would both overrun the page (4 bytes into a
// 2-byte slot) and hand the consumers a bit pattern they read as an integer, so the two
// writers have to agree.
// =============================================================================

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// The Q14 scale is folded into the projection matrix by the reader, so this kernel never
// converts a normalised coordinate — it receives Q14 units and only removes the offset.
// The saturation bounds are still its job: +-2 in grid units is +-32768 here.
constexpr float GRID_Q14_OFFSET = 16384.0f;   // the -1 the normalise leaves, in Q14
constexpr float GRID_Q14_MIN = -32768.0f;
constexpr float GRID_Q14_MAX = 32767.0f;

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


    // Load image_wh into L1
    uint32_t wh_l1_addr = get_write_ptr(cb_wh_scratch);
    for (uint32_t page = 0; page < NC; page++) {
        uint64_t noc_addr = wh_accessor.get_noc_addr(page);
        noc_async_read(noc_addr, wh_l1_addr + page * wh_page_size, wh_page_size);
    }
    noc_async_read_barrier();

    // Process each anchor × camera projected tile
    for (uint32_t a_idx = 0; a_idx < num_anchors; a_idx++) {
        uint32_t a = anchor_offset + a_idx;

        for (uint32_t c = 0; c < NC; c++) {
            // Wait for projected tile from compute
            cb_wait_front(cb_out, 1);
            volatile tt_l1_ptr uint32_t* proj_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out));

            // image_wh is no longer read per point — the reader folds 2/W and 2/H into the
            // projection matrix. The pages stay resident because the output scratch is laid
            // out after them.
            uint32_t out_scratch = wh_l1_addr + NC * wh_page_size;
            volatile tt_l1_ptr int16_t* out = reinterpret_cast<volatile tt_l1_ptr int16_t*>(out_scratch);

            // The compute kernel now delivers the grid coordinate already divided by depth
            // and already in Q14 units, less the -1 offset that the normalise leaves.
            // Columns 1 and 2, because column 0 carries depth so the SFPU broadcast could
            // reach it. All that is left here is that offset, the clamp and the cast —
            // the perspective divide and the normalise, which measured 73% of this path,
            // are gone.
            for (uint32_t p = 0; p < NUM_PTS; p++) {
                float qx = tile_get_f32(proj_tile, p, 1) - GRID_Q14_OFFSET;
                float qy = tile_get_f32(proj_tile, p, 2) - GRID_Q14_OFFSET;

                if (qx < GRID_Q14_MIN) qx = GRID_Q14_MIN;
                if (qx > GRID_Q14_MAX) qx = GRID_Q14_MAX;
                if (qy < GRID_Q14_MIN) qy = GRID_Q14_MIN;
                if (qy > GRID_Q14_MAX) qy = GRID_Q14_MAX;

                out[p * 2 + 0] = (int16_t)(int32_t)(qx >= 0.0f ? qx + 0.5f : qx - 0.5f);
                out[p * 2 + 1] = (int16_t)(int32_t)(qy >= 0.0f ? qy + 0.5f : qy - 0.5f);
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
