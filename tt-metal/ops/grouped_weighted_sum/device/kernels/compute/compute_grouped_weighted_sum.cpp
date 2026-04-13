// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Compute: TILE mode = direct bcast_cols. RM mode = tilize + bcast_cols.

#include <cstdint>
#include "api/compile_time_args.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    constexpr uint32_t feat_cb   = get_compile_time_arg_val(0);
    constexpr uint32_t wt_cb    = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb    = get_compile_time_arg_val(2);
    constexpr uint32_t G         = get_compile_time_arg_val(3);
    constexpr uint32_t RM_MODE   = get_compile_time_arg_val(4);
    constexpr uint32_t tile_cb   = get_compile_time_arg_val(5);

    uint32_t num_wus     = get_arg_val<uint32_t>(0);
    uint32_t chunk_size  = get_arg_val<uint32_t>(1);
    uint32_t total_clp   = get_arg_val<uint32_t>(2);

    if constexpr (RM_MODE) {
        // Initialize tilize pipeline first
        tilize_init(feat_cb, G, tile_cb);

        for (uint32_t wu = 0; wu < num_wus; wu++) {
            cb_reserve_back(out_cb, G);

            for (uint32_t clp = 0; clp < chunk_size; clp++) {
                // 1. Tilize RM→TILE
                cb_wait_front(feat_cb, G);  // G pages = 16KB of RM data
                cb_reserve_back(tile_cb, G);
                tilize_block(feat_cb, G, tile_cb);
                cb_push_back(tile_cb, G);
                cb_pop_front(feat_cb, G);

                // 2. Switch to bcast mode
                tilize_uninit(feat_cb, tile_cb);
                binary_op_init_common(tile_cb, wt_cb, out_cb);
                mul_bcast_cols_init_short(tile_cb, wt_cb);

                // 3. Multiply + accumulate
                cb_wait_front(tile_cb, G);
                cb_wait_front(wt_cb, G);
                pack_reconfig_l1_acc(clp > 0 ? 1 : 0);

                for (uint32_t g = 0; g < G; g++) {
                    tile_regs_acquire();
                    mul_tiles_bcast_cols(tile_cb, wt_cb, g, g, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile<true>(0, out_cb, g);
                    tile_regs_release();
                }

                cb_pop_front(tile_cb, G);
                cb_pop_front(wt_cb, G);

                // 4. Switch back to tilize for next CLP
                tilize_init(feat_cb, G, tile_cb);
            }

            pack_reconfig_l1_acc(0);
            cb_push_back(out_cb, G);
        }
    } else {
        // TILE mode: original path
        binary_op_init_common(feat_cb, wt_cb, out_cb);
        mul_bcast_cols_init_short(feat_cb, wt_cb);

        for (uint32_t wu = 0; wu < num_wus; wu++) {
            cb_reserve_back(out_cb, G);

            for (uint32_t clp = 0; clp < chunk_size; clp++) {
                cb_wait_front(feat_cb, G);
                cb_wait_front(wt_cb, G);
                pack_reconfig_l1_acc(clp > 0 ? 1 : 0);

                for (uint32_t g = 0; g < G; g++) {
                    tile_regs_acquire();
                    mul_tiles_bcast_cols(feat_cb, wt_cb, g, g, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile<true>(0, out_cb, g);
                    tile_regs_release();
                }

                cb_pop_front(feat_cb, G);
                cb_pop_front(wt_cb, G);
            }

            pack_reconfig_l1_acc(0);
            cb_push_back(out_cb, G);
        }
    }
}
