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
    // SKIP_MODE: the reader decides per work unit how many clp iterations are
    // live (dead-camera rows are skipped) and promises the count in a header
    // page. Control flow here runs identically on all three TRISCs, so a plain
    // L1 read of the count is race-free — the same mechanism get_arg_val uses.
    constexpr uint32_t SKIP_MODE = get_compile_time_arg_val(6);

    uint32_t num_wus     = get_arg_val<uint32_t>(0);
    uint32_t chunk_size  = get_arg_val<uint32_t>(1);
    uint32_t total_clp   = get_arg_val<uint32_t>(2);
    uint32_t mbox_nonce  = get_arg_val<uint32_t>(3);
    // Absolute L1 address of the per-core mailbox (an L1-sharded tensor whose
    // shards sit at the SAME address on every core). MATH has no cb_interface
    // at all in firmware, so a CB cannot carry this — a raw address can: every
    // TRISC is a RISC-V core with plain L1 loads.
    uint32_t mbox_addr   = get_arg_val<uint32_t>(4);

    if constexpr (RM_MODE) {
        // Configure BOTH pipelines up front. The loop below alternates between tilize and
        // bcast, and without this the first iteration ran the bcast half with whatever the
        // PREVIOUSLY EXECUTED OP left in the unpacker/packer config. That made gws correct
        // on every repeat call but wrong on the first call after any other op had run —
        // silently, and only in RM_MODE. See debug/verify_gws_determinism.py.
        binary_op_init_common(tile_cb, wt_cb, out_cb);
        mul_bcast_cols_init_short(tile_cb, wt_cb);
        tilize_init(feat_cb, G, tile_cb);

        for (uint32_t wu = 0; wu < num_wus; wu++) {
            cb_reserve_back(out_cb, G);

            uint32_t iters = chunk_size;
            if constexpr (SKIP_MODE) {
                // all three TRISC threads execute this control flow; each
                // spins until the reader tags this work unit's slot
                volatile uint32_t* mbox = (volatile uint32_t*)mbox_addr;
                const uint32_t slot = (wu & 3u) * 2u;
                while (mbox[slot] != mbox_nonce + wu) {
                }
                iters = mbox[slot + 1];
            }
            for (uint32_t clp = 0; clp < iters; clp++) {
                // 1. Tilize RM→TILE.
                // L1 accumulate must be OFF here: tile_cb is only G pages deep, so every
                // iteration packs to the same L1 address. With acc still enabled from the
                // previous iteration's output pack, the tilized features would be summed
                // into the previous iteration's, making clp i contribute i times.
                pack_reconfig_l1_acc(0);
                tilize_init(feat_cb, G, tile_cb);
                cb_wait_front(feat_cb, G);  // G pages = 16KB of RM data
                cb_reserve_back(tile_cb, G);
                tilize_block(feat_cb, G, tile_cb);
                cb_push_back(tile_cb, G);
                cb_pop_front(feat_cb, G);

                // 2. Switch back to bcast. Only the short init is needed: the full
                // hw configure was already done once before the loop.
                tilize_uninit(feat_cb, tile_cb);
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

            uint32_t iters = chunk_size;
            if constexpr (SKIP_MODE) {
                // all three TRISC threads execute this control flow; each
                // spins until the reader tags this work unit's slot
                volatile uint32_t* mbox = (volatile uint32_t*)mbox_addr;
                const uint32_t slot = (wu & 3u) * 2u;
                while (mbox[slot] != mbox_nonce + wu) {
                }
                iters = mbox[slot + 1];
            }
            for (uint32_t clp = 0; clp < iters; clp++) {
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
