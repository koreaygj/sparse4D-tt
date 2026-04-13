// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Compute kernel for fused KPS rotation + projection
//
// Per anchor (one tile at a time):
//   1. matmul_tiles: key_points [32,32] × rot_matrix [32,32] → rotated [32,32]  (bf16)
//   2. add_tiles: rotated + center_broadcast → translated [32,32]                (bf16)
//   For each camera:
//     3. add_tiles: translated + ones_col3 → pts_homo [32,32]                    (f32)
//     4. matmul_tiles: pts_homo [32,32] × proj_T [32,32] → projected [32,32]     (f32)
//     5. Push projected tile to output CB (writer does perspective divide + normalize)
// =============================================================================

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

void kernel_main() {
    uint32_t num_anchors = get_arg_val<uint32_t>(0);
    uint32_t NC          = get_arg_val<uint32_t>(1);

    // Circular buffer indices (must match program factory)
    constexpr auto cb_kp       = tt::CBIndex::c_0;   // key_points tile [32,32] bf16
    constexpr auto cb_rot      = tt::CBIndex::c_1;   // rotation matrix tile [32,32] bf16
    constexpr auto cb_center   = tt::CBIndex::c_2;   // center broadcast tile [32,32] bf16
    constexpr auto cb_ones     = tt::CBIndex::c_3;   // ones at col3 tile [32,32] f32
    constexpr auto cb_proj     = tt::CBIndex::c_4;   // proj_T tile [32,32] f32 (per camera)
    constexpr auto cb_rotated  = tt::CBIndex::c_5;   // intermediate: rotated bf16
    constexpr auto cb_trans    = tt::CBIndex::c_6;    // intermediate: translated bf16
    constexpr auto cb_homo     = tt::CBIndex::c_7;    // intermediate: homogeneous f32
    constexpr auto cb_out      = tt::CBIndex::c_16;   // projected output f32

    constexpr uint32_t dst0 = 0;

    for (uint32_t a = 0; a < num_anchors; a++) {
        // ---- Step 1: Rotation via matmul ----
        // rotated = key_points × rot_matrix
        cb_wait_front(cb_kp, 1);
        cb_wait_front(cb_rot, 1);

        tile_regs_acquire();
        mm_init(cb_kp, cb_rot, cb_rotated);
        matmul_tiles(cb_kp, cb_rot, 0, 0, dst0);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_rotated, 1);
        pack_tile(dst0, cb_rotated);  // Pack as bf16 → natural bf16 truncation!
        tile_regs_release();
        cb_push_back(cb_rotated, 1);

        cb_pop_front(cb_kp, 1);
        cb_pop_front(cb_rot, 1);

        // ---- Step 2: Translation via add ----
        // translated = rotated + center_broadcast
        cb_wait_front(cb_rotated, 1);
        cb_wait_front(cb_center, 1);

        tile_regs_acquire();
        binary_op_init_common(cb_rotated, cb_center, cb_trans);
        add_tiles_init(cb_rotated, cb_center);
        add_tiles(cb_rotated, cb_center, 0, 0, dst0);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_trans, 1);
        pack_tile(dst0, cb_trans);  // Pack as bf16 → truncation matches Python
        tile_regs_release();
        cb_push_back(cb_trans, 1);

        cb_pop_front(cb_rotated, 1);
        cb_pop_front(cb_center, 1);

        // ---- Per camera: projection ----
        for (uint32_t c = 0; c < NC; c++) {
            // Step 3: Add ones at column 3 for homogeneous coords
            // pts_homo = translated + ones_col3
            cb_wait_front(cb_trans, 1);
            cb_wait_front(cb_ones, 1);

            tile_regs_acquire();
            binary_op_init_common(cb_trans, cb_ones, cb_homo);
            add_tiles_init(cb_trans, cb_ones);
            add_tiles(cb_trans, cb_ones, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_homo, 1);
            pack_tile(dst0, cb_homo);  // f32
            tile_regs_release();
            cb_push_back(cb_homo, 1);

            // Don't pop cb_trans yet — reuse for next camera
            // Don't pop cb_ones — constant, reused

            // Step 4: Projection matmul
            // projected = pts_homo × proj_T
            cb_wait_front(cb_homo, 1);
            cb_wait_front(cb_proj, 1);

            tile_regs_acquire();
            mm_init(cb_homo, cb_proj, cb_out);
            matmul_tiles(cb_homo, cb_proj, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(dst0, cb_out);  // f32 projected result
            tile_regs_release();
            cb_push_back(cb_out, 1);

            cb_pop_front(cb_homo, 1);
            cb_pop_front(cb_proj, 1);
        }

        // Pop shared resources after all cameras done
        if (NC > 0) {
            cb_pop_front(cb_trans, 1);
            // cb_ones is constant — popped and re-pushed by reader once
        }
    }
}
