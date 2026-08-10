// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Reader: TILE features (original) + RM features (reads 32 sticks per tile-row)
// RM mode: reads 32 sticks at stride 1, pushes as FEAT_TC "pages" to feat_cb
// Compute tilizes the RM data and processes as TILE.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t feat_addr     = get_arg_val<uint32_t>(0);
    uint32_t wt_addr       = get_arg_val<uint32_t>(1);
    uint32_t num_wus       = get_arg_val<uint32_t>(2);
    uint32_t start_wu      = get_arg_val<uint32_t>(3);
    uint32_t N_TR_TOTAL    = get_arg_val<uint32_t>(4);
    uint32_t chunk_size    = get_arg_val<uint32_t>(5);
    uint32_t total_clp     = get_arg_val<uint32_t>(6);

    constexpr uint32_t feat_cb        = get_compile_time_arg_val(0);
    constexpr uint32_t wt_raw_cb      = get_compile_time_arg_val(1);
    constexpr uint32_t wt_col_cb      = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_GROUPS     = get_compile_time_arg_val(3);
    constexpr uint32_t FEAT_TC        = get_compile_time_arg_val(4);
    constexpr uint32_t feat_page_bytes = get_compile_time_arg_val(5);  // tile_bytes for TILE, stick_bytes for RM
    constexpr uint32_t wt_tile_bytes  = get_compile_time_arg_val(6);
    constexpr uint32_t N_TR           = get_compile_time_arg_val(7);

    constexpr auto feat_args = TensorAccessorArgs<8>();
    constexpr auto wt_args = TensorAccessorArgs<feat_args.next_compile_time_args_offset()>();

    constexpr uint32_t RM_OFFSET = wt_args.next_compile_time_args_offset();
    constexpr uint32_t RM_MODE     = get_compile_time_arg_val(RM_OFFSET);
    constexpr uint32_t RM_N        = get_compile_time_arg_val(RM_OFFSET + 1);  // total anchors
    constexpr uint32_t RM_STICK_SZ = get_compile_time_arg_val(RM_OFFSET + 2);  // C*2
    // COMPACT weights: [anchors, clp*G] instead of [clp, anchors, G]. The second layout
    // spends 24 of every tile's 32 columns on padding, because G is 8 and a tile is 32
    // wide; the first fills every column, so a tile covers 32/G consecutive clp at once
    // and the whole tensor is 4x smaller. Getting there also costs the model nothing: the
    // linear already produces [anchors, clp*G], and it is the reshape INTO [anchors, clp, G]
    // that was the single most expensive op in the attention-weight path.
    constexpr uint32_t WT_COMPACT = get_compile_time_arg_val(RM_OFFSET + 3);
    constexpr uint32_t WT_TC      = get_compile_time_arg_val(RM_OFFSET + 4);  // tiles per row

    // Feature accessor: use correct page_size based on mode
    const auto feat_acc = TensorAccessor(feat_args, feat_addr, feat_page_bytes);
    const auto wt_acc = TensorAccessor(wt_args, wt_addr, wt_tile_bytes);

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t feat_tpb = N_TR * FEAT_TC;  // tiles per batch (TILE mode only)
    constexpr uint32_t wt_tpb = N_TR;

    // The last tile row covers anchors past the end of the tensor (900..927 of 928), which
    // the loop below never reads. tilize still consumes all 32 rows, so zero the buffer once
    // here rather than leaving stale L1: those rows are sliced off the output, but they must
    // not be able to carry a NaN into the accumulation. Once is enough — nothing ever writes
    // to the tail sticks, so they stay zero for the life of the kernel. Doing this per
    // iteration instead cost 3.4 ms/frame, since it re-zeroed the same addresses 78 times.
    if constexpr (RM_MODE) {
        volatile tt_l1_ptr uint32_t* z =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(feat_cb));
        for (uint32_t i = 0; i < (NUM_GROUPS * 2 * feat_page_bytes) / 4; i++) {
            z[i] = 0;
        }
    }

    // wt_raw_cb holds a single page that is never pushed or popped, so its address is
    // fixed and a tile already sitting there can be reused. In the compact layout the
    // inner loop walks clp one at a time while the tile only changes every 32/G steps.
    uint32_t cached_wt_tile = 0xFFFFFFFFu;

    for (uint32_t wu = 0; wu < num_wus; wu++) {
        uint32_t wu_id = start_wu + wu;
        uint32_t n_tr = wu_id % N_TR_TOTAL;
        uint32_t chunk_id = wu_id / N_TR_TOTAL;

        uint32_t clp_start = chunk_id * chunk_size;
        uint32_t clp_end = clp_start + chunk_size;
        if (clp_end > total_clp) clp_end = total_clp;

        for (uint32_t clp = clp_start; clp < clp_end; clp++) {
            // === Feature read ===
            cb_reserve_back(feat_cb, FEAT_TC);
            uint32_t feat_l1 = get_write_ptr(feat_cb);

            if constexpr (RM_MODE) {
                // RM: read 32 sticks (1 tile-row) into feat_cb as contiguous RM data
                // Page_id in RM tensor: clp * RM_N + anchor
                uint32_t anchor_start = n_tr * TILE_H;
                for (uint32_t r = 0; r < TILE_H; r++) {
                    uint32_t anchor = anchor_start + r;
                    if (anchor < RM_N) {
                        uint32_t page_id = clp * RM_N + anchor;
                        noc_async_read(feat_acc.get_noc_addr(page_id),
                                       feat_l1 + r * RM_STICK_SZ, RM_STICK_SZ);
                    }
                }
            } else {
                // TILE: read FEAT_TC tiles
                uint32_t feat_base = clp * feat_tpb + n_tr * FEAT_TC;
                for (uint32_t fc = 0; fc < FEAT_TC; fc++) {
                    noc_async_read(feat_acc.get_noc_addr(feat_base + fc),
                                   feat_l1 + fc * feat_page_bytes, feat_page_bytes);
                }
            }

            // === Weight read ===
            // Compact: this clp's G columns start at column clp*G of tile-row n_tr, so one
            // tile serves 32/G consecutive clp and is only re-read when clp crosses it.
            uint32_t wt_tile_id, col_base;
            if constexpr (WT_COMPACT) {
                const uint32_t c0 = clp * NUM_GROUPS;
                wt_tile_id = n_tr * WT_TC + (c0 >> 5);
                col_base = c0 & 31;
            } else {
                wt_tile_id = clp * wt_tpb + n_tr;
                col_base = 0;
            }
            uint32_t wt_raw_l1 = get_write_ptr(wt_raw_cb);
            if (wt_tile_id != cached_wt_tile) {
                noc_async_read(wt_acc.get_noc_addr(wt_tile_id), wt_raw_l1, wt_tile_bytes);
                cached_wt_tile = wt_tile_id;
            }
            noc_async_read_barrier();
            cb_push_back(feat_cb, FEAT_TC);

            // === Extract weight columns ===
            // A 32x32 tile is stored as four 16x16 faces, so a column lands in face
            // (row>=16)*2 + (col>=16) at offset (row%16)*16 + col%16. col_base is 0 in the
            // non-compact layout, which reduces this to the original "column g" case.
            volatile tt_l1_ptr uint16_t* raw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wt_raw_l1);
            cb_reserve_back(wt_col_cb, NUM_GROUPS);
            uint32_t col_l1 = get_write_ptr(wt_col_cb);
            for (uint32_t g = 0; g < NUM_GROUPS; g++) {
                const uint32_t sc = col_base + g;
                const uint32_t cf = (sc >= 16) ? 1 : 0;
                const uint32_t cc = sc & 15;
                volatile tt_l1_ptr uint16_t* col = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                    col_l1 + g * wt_tile_bytes);
                for (uint32_t r = 0; r < 32; r++) {
                    uint32_t sf = ((r >= 16) ? 2 : 0) + cf;
                    uint32_t fr = r % 16;
                    uint16_t val = raw[sf * 256 + fr * 16 + cc];
                    uint32_t df = (r >= 16) ? 2 : 0;
                    col[df * 256 + fr * 16 + 0] = val;
                }
            }
            cb_push_back(wt_col_cb, NUM_GROUPS);
        }
    }
}
