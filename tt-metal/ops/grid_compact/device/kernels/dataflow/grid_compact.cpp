// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Grid compaction.
//
// Walks every (camera, anchor) row of the DFA sampling grid and keeps only the rows
// with at least one point inside the image. ~80% of rows project entirely outside
// their camera; grid_sample skips their DRAM reads but still pays CB page, barrier
// and reduce for each one, so removing them up front is what actually saves time.
//
// The grid is Q14 fixed point (see kps_project_fused), so the bounds test is a plain
// integer |v| < thr on int16 — no bit tricks and no soft-float compare, which on an
// FPU-less dataflow core would be ~25 cycles per point.
//
// The two axes get their own threshold. With align_corners=False a point contributes
// iff g is in [-1 - 1/S, 1 + 1/S), S being W on x and H on y -- and the coarsest FPN
// level is 8 x 22, so one shared threshold would have to use 1 + 1/8 on BOTH axes and
// would keep everything.
//
// TWO PHASES ACROSS CORES. The bounds test is the bulk of the work and has no ordering
// dependency, so every core tests its own slice of the rows; the writes need a running
// output counter, so one core does them. Measured on the production shape (2700 rows,
// 13 points, ~80% dropped) the single-core version spent ~440 us of its 763 us in the
// test alone, which is what phase 1 removes.
//
//   phase 1  core i tests rows [i*RPC, (i+1)*RPC) and leaves one byte per row in its
//            slice of a shared mask buffer, then relays that slice into core 0's copy
//            and bumps core 0's semaphore.
//   phase 2  core 0 waits for all NUM_CORES-1 relays, then walks the mask in row order
//            and emits the kept rows.
//
// Splitting the WRITES too would need a prefix sum over the per-core counts. It is
// worth doing only after phase 2 stops dominating; the mask is already the hard part
// of that change, since every core can derive its own slot range from it.
//
// Rows past the kept count are left untouched in cgrid — their index entry is
// SENTINEL, and transposed_s2i skips those, so stale coordinates can never reach
// the output. That avoids writing padding rows every call.
//
// Two output layouts. PER CAMERA (no bidx) puts camera c's kept rows in
// cgrid[c*CAP .. c*CAP+CAP), which grid_sample can read unmodified because it derives a
// row's source image from its POSITION. POOLED (bidx given) puts every camera's rows in
// one shared list and records each row's camera, which needs grid_sample's batch_index.
//
// Pooled is what makes compaction pay. Per camera the budget must cover the busiest
// CAMERA, and the cameras do not peak together: measured over 16 scenes, the zero-loss
// budget is 3 x 563 = 1689 rows per camera against 902 pooled, for the same guarantee.
//
// A third output, `flags`, marks the kept (camera, anchor) pairs with 1.0. Dropped
// rows are never written into the feature buffer, so it keeps LAST frame's values
// there; multiplying the attention weights by these flags annihilates that stale
// data. Zeroing the 68.6 MB feature buffer instead would cost more than compaction
// saves.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

constexpr uint32_t NUM_ROWS  = get_compile_time_arg_val(0);
constexpr uint32_t NUM_PTS   = get_compile_time_arg_val(1);
constexpr uint32_t ROW_W     = get_compile_time_arg_val(2);
constexpr uint32_t CAP       = get_compile_time_arg_val(3);
constexpr uint32_t ANCHORS   = get_compile_time_arg_val(4);
constexpr uint32_t THR_X     = get_compile_time_arg_val(5);
constexpr uint32_t ROW_CB    = get_compile_time_arg_val(6);
constexpr uint32_t IDX_CB    = get_compile_time_arg_val(7);
constexpr uint32_t BATCH     = get_compile_time_arg_val(8);
constexpr uint32_t FLG_CB    = get_compile_time_arg_val(9);
constexpr uint32_t FLG_W     = get_compile_time_arg_val(10);
constexpr uint32_t THR_Y     = get_compile_time_arg_val(11);
// Pooled: one shared list of kept rows, each carrying its camera in bidx, instead of a
// fixed block per camera. The cameras do not peak together, so the shared list needs
// barely half the capacity for the same zero-loss guarantee.
constexpr uint32_t POOLED    = get_compile_time_arg_val(12);
constexpr uint32_t BIDX_CB   = get_compile_time_arg_val(13);
constexpr uint32_t BIDX_W    = get_compile_time_arg_val(14);
constexpr uint32_t NUM_CORES = get_compile_time_arg_val(15);
constexpr uint32_t RPC       = get_compile_time_arg_val(16);  // rows per core, phase 1
constexpr uint32_t MASK_BLK  = get_compile_time_arg_val(17);  // bytes per core in the mask
constexpr uint32_t MASK_CB   = get_compile_time_arg_val(18);
constexpr uint32_t SEM_ID    = get_compile_time_arg_val(19);
constexpr uint32_t C0_X      = get_compile_time_arg_val(20);  // physical NOC coords of core 0
constexpr uint32_t C0_Y      = get_compile_time_arg_val(21);

constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
constexpr uint16_t BF16_ONE = 0x3F80u;  // 1.0f truncated to bfloat16 (flags are bf16)
constexpr uint32_t row_bytes = ROW_W * 2;  // int16 fixed-point grid
// NOC endpoints must be 32 B aligned, and a row is 26 floats = 104 B, so the L1
// batch buffer is strided at the aligned size while only row_bytes is transferred.
// Packing rows tight in L1 put every odd row at a misaligned address and the reads
// came back as garbage — which read as in bounds and kept everything.
constexpr uint32_t row_stride = ((row_bytes + 31) / 32) * 32;
constexpr uint32_t NUM_CAMS = NUM_ROWS / ANCHORS;
constexpr uint32_t idx_bytes = (POOLED ? CAP : NUM_CAMS * CAP) * 4;
constexpr uint32_t flag_bytes = FLG_W * 2;
constexpr uint32_t flag_stride = ((flag_bytes + 31) / 32) * 32;  // same, per camera
constexpr uint32_t bidx_row_bytes = BIDX_W * 4;

// The accessor args start after ALL of the scalars above. Adding a compile-time arg
// ahead of this offset without moving it silently corrupts every accessor.
constexpr auto grid_args  = TensorAccessorArgs<22>();
constexpr auto cgrid_args = TensorAccessorArgs<grid_args.next_compile_time_args_offset()>();
constexpr auto index_args = TensorAccessorArgs<cgrid_args.next_compile_time_args_offset()>();
constexpr auto flags_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
constexpr uint32_t BIDX_OFF = flags_args.next_compile_time_args_offset();

// bidx's accessor args only EXIST in pooled mode, so everything that touches them has to
// live in a template: an `if constexpr` inside kernel_main() would not discard the branch
// (kernel_main is not a template) and TensorAccessorArgs<BIDX_OFF> would still be
// instantiated past the end of the arg list. It has to stay a PARTIAL specialization for
// the same reason — a full `template <> struct BidxWriter<true>` is a concrete class and
// its members are instantiated where it is defined, not where it is used.
template <bool P, uint32_t OFF>
struct BidxWriter {
    explicit BidxWriter(uint32_t) {}
    FORCE_INLINE void write(uint32_t, uint32_t) const {}
};

template <uint32_t OFF>
struct BidxWriter<true, OFF> {
    decltype(TensorAccessor(TensorAccessorArgs<OFF>(), 0u)) acc;
    uint32_t stage_l1;

    // One staged row per camera instead of the whole CAP x BIDX_W buffer: a row is
    // (camera, 0, 0, ...) and only NUM_CAMS distinct rows ever exist, so the L1 cost
    // drops from ~30 KB to under 256 B and the per-slot work becomes a single write.
    explicit BidxWriter(uint32_t bidx_addr)
        : acc(TensorAccessorArgs<OFF>(), bidx_addr), stage_l1(get_write_ptr(BIDX_CB)) {
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_l1);
        for (uint32_t c = 0; c < NUM_CAMS; c++) {
            for (uint32_t j = 0; j < BIDX_W; j++) {
                s[c * BIDX_W + j] = (j == 0) ? c : 0u;
            }
        }
    }

    // Every bidx slot must name a real camera, including the ones past the kept count:
    // grid_sample still walks those sticks and turns the value into a NOC address, so a
    // stale one would be an out-of-range read rather than a discarded sample. Camera 0
    // is written into the tail for exactly that reason.
    FORCE_INLINE void write(uint32_t slot, uint32_t cam) const {
        noc_async_write(stage_l1 + cam * bidx_row_bytes, acc.get_noc_addr(slot), bidx_row_bytes);
    }
};

void kernel_main() {
    // The buffer addresses are the same on every core, so they go in the COMMON args and
    // are the only thing the host rewrites per dispatch. Putting them in per-core args
    // meant re-writing 5 words x 64 cores every call, which measured 84 us of host time.
    const uint32_t grid_addr  = get_common_arg_val<uint32_t>(0);
    const uint32_t cgrid_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t index_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t flags_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t bidx_addr  = get_common_arg_val<uint32_t>(4);
    const uint32_t core_id    = get_arg_val<uint32_t>(0);

    // No explicit page size: that argument is the ALIGNED page size, and DRAM pages are
    // padded to 32 B. A grid row is 26 floats = 104 B, which pads to 128, so passing
    // row_bytes here put every row after the first at the wrong address — the kernel
    // then read garbage and kept every row. Let TensorAccessorArgs supply the real one.
    const auto grid_acc = TensorAccessor(grid_args, grid_addr);

    const uint32_t row_l1  = get_write_ptr(ROW_CB);
    const uint32_t mask_l1 = get_write_ptr(MASK_CB);
    volatile tt_l1_ptr uint8_t* mask = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(mask_l1);

    // ---------------------------------------------------------------- phase 1
    // Every core tests its own contiguous slice. The slices are laid out at a stride of
    // MASK_BLK rather than RPC so each core's relay lands on a 32 B boundary, which the
    // NOC requires; the slack bytes are zeroed so phase 2 can read a block whole.
    const uint32_t my_start = core_id * RPC;
    uint32_t my_count = 0;
    if (my_start < NUM_ROWS) {
        my_count = NUM_ROWS - my_start;
        if (my_count > RPC) {
            my_count = RPC;
        }
    }

    for (uint32_t b = 0; b < my_count; b++) {
        noc_async_read(grid_acc.get_noc_addr(my_start + b), row_l1 + b * row_stride, row_bytes);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint8_t* my_mask = mask + core_id * MASK_BLK;
    for (uint32_t b = 0; b < my_count; b++) {
        volatile tt_l1_ptr int16_t* rp =
            reinterpret_cast<volatile tt_l1_ptr int16_t*>(row_l1 + b * row_stride);
        uint8_t valid = 0;
        for (uint32_t p = 0; p < NUM_PTS; p++) {
            const int32_t vx = rp[2 * p];
            const int32_t vy = rp[2 * p + 1];
            const int32_t ax = vx < 0 ? -vx : vx;
            const int32_t ay = vy < 0 ? -vy : vy;
            if (ax < (int32_t)THR_X && ay < (int32_t)THR_Y) {
                valid = 1;
                break;
            }
        }
        my_mask[b] = valid;
    }
    for (uint32_t b = my_count; b < MASK_BLK; b++) {
        my_mask[b] = 0;
    }

    Noc noc;
    Semaphore<> sem(SEM_ID);

    if (core_id != 0) {
        noc_async_write(mask_l1 + core_id * MASK_BLK,
                        get_noc_addr(C0_X, C0_Y, mask_l1 + core_id * MASK_BLK),
                        MASK_BLK);
        noc_async_write_barrier();  // the relay must land before the semaphore says it did
        sem.up(noc, C0_X, C0_Y, 1);
        return;
    }

    // ---------------------------------------------------------------- phase 2
    sem.wait(NUM_CORES - 1);

    const auto cgrid_acc = TensorAccessor(cgrid_args, cgrid_addr);
    const auto index_acc = TensorAccessor(index_args, index_addr);
    const auto flags_acc = TensorAccessor(flags_args, flags_addr);
    BidxWriter<POOLED != 0, BIDX_OFF> bidx(bidx_addr);

    const uint32_t idx_l1 = get_write_ptr(IDX_CB);
    const uint32_t flg_l1 = get_write_ptr(FLG_CB);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    // One bf16 flag per (camera, anchor), kept in L1 for the whole pass and dumped at
    // the end: a per-camera flush would have to barrier mid-batch before reusing the
    // buffer. NUM_CAMS * FLG_W * 2 B is ~5 KB, so there is no reason to be clever.
    volatile tt_l1_ptr uint16_t* flg = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(flg_l1);
    for (uint32_t i = 0; i < NUM_CAMS * (flag_stride / 2); i++) {
        flg[i] = 0;
    }

    uint32_t pend_row[BATCH];
    uint32_t pend_cam[BATCH];
    uint32_t np = 0;
    uint32_t n = 0;        // kept rows in the current camera (pooled: overall)
    uint32_t cur_cam = 0;  // camera of the rows being written
    uint32_t scan_cam = 0;  // camera of the row being scanned, always >= cur_cam

    // Rows are fetched BATCH at a time. One barrier per batch instead of per row:
    // a per-row barrier waits out the full DRAM round trip before issuing the next
    // read, which measured 1.96 ms/call — the loop was latency-bound, not busy.
    // Only the KEPT rows are fetched here; phase 1 already read everything once, but
    // re-reading ~900 of 2700 rows is cheaper than staging 345 KB across cores.
    auto flush = [&]() {
        for (uint32_t i = 0; i < np; i++) {
            noc_async_read(grid_acc.get_noc_addr(pend_row[i]), row_l1 + i * row_stride, row_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < np; i++) {
            const uint32_t r = pend_row[i];
            const uint32_t c = pend_cam[i];
            if constexpr (!POOLED) {
                if (c != cur_cam) {
                    while (n < CAP) {  // SENTINEL-fill the tail of the camera we just left
                        idx[cur_cam * CAP + n] = SENTINEL;
                        n++;
                    }
                    cur_cam = c;
                    n = 0;
                }
            }
            if (n >= CAP) {
                continue;
            }
            const uint32_t slot = POOLED ? n : (c * CAP + n);
            noc_async_write(row_l1 + i * row_stride, cgrid_acc.get_noc_addr(slot), row_bytes);
            idx[slot] = r;
            bidx.write(slot, c);
            // Flag the rows that were actually KEPT, not the ones that merely passed the
            // bounds test: a row past CAP is dropped too, and its slot in the feature
            // buffer keeps last frame's values.
            flg[c * (flag_stride / 2) + (r - c * ANCHORS)] = BF16_ONE;
            n++;
        }
        // the batch buffer is about to be refilled, so the writes out of it must land
        noc_async_write_barrier();
        np = 0;
    };

    bool done = false;
    for (uint32_t blk = 0; blk < NUM_CORES && !done; blk++) {
        const uint32_t base = blk * RPC;
        if (base >= NUM_ROWS) {
            break;
        }
        uint32_t cnt = NUM_ROWS - base;
        if (cnt > RPC) {
            cnt = RPC;
        }
        volatile tt_l1_ptr uint8_t* blk_mask = mask + blk * MASK_BLK;
        for (uint32_t j = 0; j < cnt; j++) {
            if (!blk_mask[j]) {
                continue;
            }
            const uint32_t r = base + j;
            // Once the budget is full there is nothing left to emit, and every row past
            // that point would still be FETCHED by flush() before being discarded. On a
            // grid where most rows pass the bounds test that was 2700 DRAM reads to write
            // 928 rows.
            if (n + np >= CAP) {
                if constexpr (POOLED) {
                    done = true;
                    break;
                }
                // Per camera the budget is per block, so only the saturated camera's
                // remaining rows are skipped — the next camera starts a fresh count.
                while (r >= (scan_cam + 1) * ANCHORS) {
                    scan_cam++;
                }
                if (scan_cam == cur_cam) {
                    continue;
                }
            }
            pend_row[np] = r;
            // Rows are camera-major and r only increases, so the camera is a running
            // counter rather than a divide — the dataflow core has no divider.
            while (r >= (scan_cam + 1) * ANCHORS) {
                scan_cam++;
            }
            pend_cam[np] = scan_cam;
            np++;
            if (np == BATCH) {
                flush();
            }
        }
    }
    if (np > 0) {
        flush();
    }

    if constexpr (POOLED) {
        for (uint32_t j = n; j < CAP; j++) {
            idx[j] = SENTINEL;
        }
    } else {
        for (uint32_t j = n; j < CAP; j++) {  // tail of the last camera
            idx[cur_cam * CAP + j] = SENTINEL;
        }
        for (uint32_t c = cur_cam + 1; c < NUM_CAMS; c++) {  // cameras that kept nothing
            for (uint32_t j = 0; j < CAP; j++) {
                idx[c * CAP + j] = SENTINEL;
            }
        }
    }
    // The tail slots must still name a real camera; the staged row 0 does that.
    for (uint32_t j = n; j < CAP; j++) {
        bidx.write(j, 0);
    }

    noc_async_write(idx_l1, index_acc.get_noc_addr(0), idx_bytes);
    for (uint32_t c = 0; c < NUM_CAMS; c++) {  // one page per camera
        noc_async_write(flg_l1 + c * flag_stride, flags_acc.get_noc_addr(c), flag_bytes);
    }
    noc_async_write_barrier();
}
