// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Grid compaction, single core.
//
// Walks every (camera, anchor) row of the DFA sampling grid and keeps only the rows
// with at least one point inside the image. ~80% of rows project entirely outside
// their camera; grid_sample skips their DRAM reads but still pays CB page, barrier
// and reduce for each one, so removing them up front is what actually saves time.
//
// Bounds test is done on raw bits: for IEEE754, |v| < thr (thr > 0) is exactly
// (bits(v) & 0x7FFFFFFF) < bits(thr). Two integer ops instead of a soft-float
// compare, which on an FPU-less dataflow core is ~25 cycles.
//
// The two axes get their own threshold. With align_corners=False a point contributes
// iff g is in [-1 - 1/S, 1 + 1/S), S being W on x and H on y -- and the coarsest FPN
// level is 8 x 22, so one shared threshold would have to use 1 + 1/8 on BOTH axes and
// would keep everything.
//
// Rows past the kept count are left untouched in cgrid — their index entry is
// SENTINEL, and transposed_s2i skips those, so stale coordinates can never reach
// the output. That avoids writing padding rows every call.
//
// Compaction is PER CAMERA: camera c's kept rows land in cgrid[c*CAP .. c*CAP+CAP).
// Pooling all cameras into one list compacts better (2.4x vs 1.5x), but grid_sample
// derives the source image from a row's POSITION (curr_batch advances every grid_hw
// sticks), so a pooled list would silently sample the wrong camera. Keeping the
// per-camera blocks preserves that mapping with no change to grid_sample at all.
// CAP must therefore cover the BUSIEST camera, measured max 62.7% of 900.
//
// A third output, `flags`, marks the kept (camera, anchor) pairs with 1.0. Dropped
// rows are never written into the feature buffer, so it keeps LAST frame's values
// there; multiplying the attention weights by these flags annihilates that stale
// data. Zeroing the 68.6 MB feature buffer instead would cost more than compaction
// saves.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// bidx's accessor args only EXIST in pooled mode, so this has to be a template: an
// `if constexpr` inside kernel_main() would not discard the branch (kernel_main is not a
// template) and TensorAccessorArgs<OFF> would still be instantiated past the arg list.
template <bool POOLED, uint32_t OFF, uint32_t BIDX_CB, uint32_t CAP, uint32_t BIDX_W>
FORCE_INLINE void write_bidx(uint32_t bidx_addr) {
    if constexpr (POOLED) {
        constexpr auto bidx_args = TensorAccessorArgs<OFF>();
        const auto bidx_acc = TensorAccessor(bidx_args, bidx_addr);
        const uint32_t bidx_l1 = get_write_ptr(BIDX_CB);
        for (uint32_t j = 0; j < CAP; j++) {  // one page per row
            noc_async_write(bidx_l1 + j * BIDX_W * 4, bidx_acc.get_noc_addr(j), BIDX_W * 4);
        }
    }
}

void kernel_main() {
    const uint32_t grid_addr  = get_arg_val<uint32_t>(0);
    const uint32_t cgrid_addr = get_arg_val<uint32_t>(1);
    const uint32_t index_addr = get_arg_val<uint32_t>(2);
    const uint32_t flags_addr = get_arg_val<uint32_t>(3);
    const uint32_t bidx_addr  = get_arg_val<uint32_t>(4);

    constexpr uint32_t NUM_ROWS = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_PTS  = get_compile_time_arg_val(1);
    constexpr uint32_t ROW_W    = get_compile_time_arg_val(2);
    constexpr uint32_t CAP      = get_compile_time_arg_val(3);
    constexpr uint32_t ANCHORS  = get_compile_time_arg_val(4);
    constexpr uint32_t THR_X    = get_compile_time_arg_val(5);
    constexpr uint32_t ROW_CB   = get_compile_time_arg_val(6);
    constexpr uint32_t IDX_CB   = get_compile_time_arg_val(7);
    constexpr uint32_t BATCH    = get_compile_time_arg_val(8);
    constexpr uint32_t FLG_CB   = get_compile_time_arg_val(9);
    constexpr uint32_t FLG_W    = get_compile_time_arg_val(10);
    constexpr uint32_t THR_Y    = get_compile_time_arg_val(11);
    // Pooled: one shared list of kept rows, each carrying its camera in bidx, instead of a
    // fixed block per camera. The cameras do not peak together, so the shared list needs
    // barely half the capacity for the same zero-loss guarantee.
    constexpr uint32_t POOLED   = get_compile_time_arg_val(12);
    constexpr uint32_t BIDX_CB  = get_compile_time_arg_val(13);
    constexpr uint32_t BIDX_W   = get_compile_time_arg_val(14);

    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    constexpr uint32_t ABS_MASK = 0x7FFFFFFFu;
    constexpr uint16_t BF16_ONE = 0x3F80u;  // 1.0f truncated to bfloat16
    constexpr uint32_t row_bytes = ROW_W * 4;
    // NOC endpoints must be 32 B aligned, and a row is 26 floats = 104 B, so the L1
    // batch buffer is strided at the aligned size while only row_bytes is transferred.
    // Packing rows tight in L1 put every odd row at a misaligned address and the reads
    // came back as garbage — which read as in bounds and kept everything.
    constexpr uint32_t row_stride = ((row_bytes + 31) / 32) * 32;
    constexpr uint32_t NUM_CAMS = NUM_ROWS / ANCHORS;
    constexpr uint32_t idx_bytes = (POOLED ? CAP : NUM_CAMS * CAP) * 4;
    constexpr uint32_t flag_bytes = FLG_W * 2;
    constexpr uint32_t flag_stride = ((flag_bytes + 31) / 32) * 32;  // same, per camera

    constexpr auto grid_args  = TensorAccessorArgs<15>();  // 0-14 are the scalars above
    constexpr auto cgrid_args = TensorAccessorArgs<grid_args.next_compile_time_args_offset()>();
    constexpr auto index_args = TensorAccessorArgs<cgrid_args.next_compile_time_args_offset()>();
    constexpr auto flags_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();
    // No explicit page size: that argument is the ALIGNED page size, and DRAM pages are
    // padded to 32 B. A grid row is 26 floats = 104 B, which pads to 128, so passing
    // row_bytes here put every row after the first at the wrong address — the kernel
    // then read garbage and kept every row. Let TensorAccessorArgs supply the real one.
    const auto grid_acc  = TensorAccessor(grid_args, grid_addr);
    const auto cgrid_acc = TensorAccessor(cgrid_args, cgrid_addr);
    const auto index_acc = TensorAccessor(index_args, index_addr);
    const auto flags_acc = TensorAccessor(flags_args, flags_addr);

    const uint32_t row_l1 = get_write_ptr(ROW_CB);
    const uint32_t idx_l1 = get_write_ptr(IDX_CB);
    const uint32_t flg_l1 = get_write_ptr(FLG_CB);
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_l1);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    // One bf16 flag per (camera, anchor), kept in L1 for the whole pass and dumped at
    // the end: a per-camera flush would have to barrier mid-batch before reusing the
    // buffer. NUM_CAMS * FLG_W * 2 B is ~5 KB, so there is no reason to be clever.
    volatile tt_l1_ptr uint16_t* flg = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(flg_l1);
    for (uint32_t i = 0; i < NUM_CAMS * (flag_stride / 2); i++) {
        flg[i] = 0;
    }
    // Every bidx slot must name a real camera, including the ones past the kept count:
    // grid_sample still walks those sticks and turns the value into a NOC address, so a
    // stale one would be an out-of-range read rather than a discarded sample.
    volatile tt_l1_ptr uint32_t* bidx = nullptr;
    if constexpr (POOLED) {
        bidx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(BIDX_CB));
        for (uint32_t i = 0; i < CAP * BIDX_W; i++) {
            bidx[i] = 0;
        }
    }

    // Rows are fetched BATCH at a time. One barrier per batch instead of per row:
    // a per-row barrier waits out the full DRAM round trip before issuing the next
    // read, which measured 1.96 ms/call — the loop was latency-bound, not busy.
    uint32_t n = 0;        // kept rows in the current camera
    uint32_t cam = 0;
    uint32_t cam_base = 0;
    uint32_t next_cam_row = ANCHORS;

    for (uint32_t base = 0; base < NUM_ROWS; base += BATCH) {
        uint32_t count = BATCH;
        if (base + count > NUM_ROWS) {
            count = NUM_ROWS - base;
        }
        for (uint32_t b = 0; b < count; b++) {
            noc_async_read(grid_acc.get_noc_addr(base + b), row_l1 + b * row_stride, row_bytes);
        }
        noc_async_read_barrier();

        for (uint32_t b = 0; b < count; b++) {
            const uint32_t r = base + b;
            if (r >= next_cam_row) {  // rows are camera-major, so this walks the cameras
                if constexpr (!POOLED) {
                    while (n < CAP) {  // SENTINEL-fill the tail of the camera we just left
                        idx[cam * CAP + n] = SENTINEL;
                        n++;
                    }
                    n = 0;
                }
                cam++;
                cam_base = next_cam_row;
                next_cam_row += ANCHORS;
            }
            volatile tt_l1_ptr uint32_t* rp = row + b * (row_stride / 4);

            bool valid = false;
            for (uint32_t p = 0; p < NUM_PTS; p++) {
                const uint32_t ax = rp[2 * p] & ABS_MASK;
                const uint32_t ay = rp[2 * p + 1] & ABS_MASK;
                if (ax < THR_X && ay < THR_Y) {
                    valid = true;
                    break;
                }
            }

            if (valid && n < CAP) {
                const uint32_t slot = POOLED ? n : (cam * CAP + n);
                if constexpr (POOLED) {
                    bidx[n * BIDX_W] = cam;
                }
                noc_async_write(row_l1 + b * row_stride,
                                cgrid_acc.get_noc_addr(slot), row_bytes);
                idx[slot] = r;
                // Flag the rows that were actually KEPT, not the ones that merely
                // passed the bounds test: a row past CAP is dropped too, and its
                // slot in the feature buffer keeps last frame's values.
                flg[cam * (flag_stride / 2) + (r - cam_base)] = BF16_ONE;
                n++;
            }
        }
        // the batch buffer is about to be refilled, so the writes out of it must land
        noc_async_write_barrier();
    }

    if constexpr (POOLED) {
        for (uint32_t j = n; j < CAP; j++) {
            idx[j] = SENTINEL;
        }
    } else {
        for (uint32_t j = n; j < CAP; j++) {  // tail of the last camera
            idx[cam * CAP + j] = SENTINEL;
        }
    }
    noc_async_write(idx_l1, index_acc.get_noc_addr(0), idx_bytes);
    for (uint32_t c = 0; c < NUM_CAMS; c++) {  // one page per camera
        noc_async_write(flg_l1 + c * flag_stride, flags_acc.get_noc_addr(c), flag_bytes);
    }
    write_bidx<POOLED != 0, flags_args.next_compile_time_args_offset(), BIDX_CB, CAP, BIDX_W>(
        bidx_addr);
    noc_async_write_barrier();
}
