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
//        with proj_T arranged so col0 = pz and col1/2 = the normalise and Q14 scale
//        already folded in (see the reader)
//     5. recip_tile + mul_tiles_bcast_cols: the perspective divide, on the SFPU
//     6. Push to output CB; the writer only subtracts the -1 offset and casts to int16
// =============================================================================

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/binary_max_min.h"

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
    constexpr auto cb_proj_raw = tt::CBIndex::c_8;    // pre-divide: col0 = pz, col1/2 = scaled x/y
    constexpr auto cb_invz     = tt::CBIndex::c_9;    // reciprocal, col0 = 1/pz
    constexpr auto cb_zfloor   = tt::CBIndex::c_10;   // constant: 1e-5 in col0, -FLT_MAX elsewhere
    constexpr auto cb_out      = tt::CBIndex::c_16;   // grid coords, Q14 before the -1

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

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
            cb_reserve_back(cb_proj_raw, 1);
            pack_tile(dst0, cb_proj_raw);  // col0 = pz, col1 = sx*px, col2 = sy*py
            tile_regs_release();
            cb_push_back(cb_proj_raw, 1);

            cb_pop_front(cb_homo, 1);
            cb_pop_front(cb_proj, 1);

            // ---- Step 5: perspective divide, on the SFPU ----
            // This is the work the writer used to do per point in soft float on a core
            // with no FPU, and it measured 73% of this path. Two tile ops replace it.
            //
            // reciprocal of the whole tile: only column 0 is meaningful (1/pz), the rest
            // is the reciprocal of numerators nobody reads.
            cb_wait_front(cb_proj_raw, 1);
            cb_wait_front(cb_zfloor, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_proj_raw);
            copy_tile(cb_proj_raw, 0, dst0);
            copy_tile_to_dst_init_short(cb_zfloor);
            copy_tile(cb_zfloor, 0, dst1);
            // Floor the depth before inverting it. A point behind the camera has pz <= 0,
            // and 1/pz would then be negative — px/pz comes back sign-flipped at a
            // plausible magnitude, which reads as a valid in-bounds sample of the wrong
            // pixel rather than an out-of-bounds one. Flooring turns it into a huge
            // coordinate that clamps out, which is what the scalar writer's z_safe did.
            binary_max_tile_init();
            binary_max_tile(dst0, dst1, dst0);
            recip_tile_init();
            recip_tile(dst0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_invz, 1);
            pack_tile(dst0, cb_invz);
            tile_regs_release();
            cb_push_back(cb_invz, 1);

            // Broadcast column 0 across the row and multiply: columns 1 and 2 become
            // sx*px/pz and sy*py/pz, i.e. the Q14 grid coordinate before the -1 offset.
            // Column 0 is the only one the COL broadcast can take, which is why the reader
            // puts depth there.
            cb_wait_front(cb_invz, 1);
            tile_regs_acquire();
            mul_bcast_cols_init_short(cb_proj_raw, cb_invz);
            mul_tiles_bcast_cols(cb_proj_raw, cb_invz, 0, 0, dst0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(dst0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);

            cb_pop_front(cb_proj_raw, 1);
            cb_pop_front(cb_invz, 1);
        }

        // Pop shared resources after all cameras done
        if (NC > 0) {
            cb_pop_front(cb_trans, 1);
            // cb_ones is constant — popped and re-pushed by reader once
        }
    }
}
