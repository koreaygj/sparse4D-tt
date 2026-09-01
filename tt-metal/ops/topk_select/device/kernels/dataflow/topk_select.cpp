// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Top-k selection over one row of scores.
//
// ttnn.topk on a [1, 900] input pads to a 32 x 928 TILE and bitonic-sorts all 32
// rows on one core — 97% of its 1.9-2.8 ms/call is sorting tile padding. This
// kernel reads only the one real row and sorts it with an LSD radix sort: all
// integer compares and moves, so the FPU-less dataflow core runs it at full rate
// (the soft-float trap only bites float ARITHMETIC; bf16 ordering is recovered
// with a bit flip and compared as uint16).
//
// Ordering contract: descending by score, ties broken by the LOWER source index
// (torch.topk semantics). The index bits ride in the low half of each record and
// are never part of a sort digit, so the stable radix passes preserve their
// original order within a tie.
//
// One core, no compute kernel. The sort is ~5k L1 word operations for N=900 —
// far below the op's dispatch cost — so fanning out across cores would only add
// synchronization for time nobody gets back.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint32_t N         = get_compile_time_arg_val(0);
    constexpr uint32_t K         = get_compile_time_arg_val(1);
    constexpr uint32_t SCORES_CB = get_compile_time_arg_val(2);
    constexpr uint32_t REC_CB    = get_compile_time_arg_val(3);
    constexpr uint32_t CNT_CB    = get_compile_time_arg_val(4);
    constexpr uint32_t VAL_CB    = get_compile_time_arg_val(5);
    constexpr uint32_t IDX_CB    = get_compile_time_arg_val(6);
    constexpr uint32_t NT        = (N + 31) / 32;  // input tiles along the row

    constexpr auto scores_args = TensorAccessorArgs<7>();
    constexpr auto values_args =
        TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto index_args =
        TensorAccessorArgs<values_args.next_compile_time_args_offset()>();

    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t values_addr = get_arg_val<uint32_t>(1);
    const uint32_t index_addr  = get_arg_val<uint32_t>(2);

    constexpr uint32_t TILE_BYTES = 32 * 32 * 2;  // bf16 tile page
    const auto scores_ta = TensorAccessor(scores_args, scores_addr, TILE_BYTES);
    const auto values_ta = TensorAccessor(values_args, values_addr, K * 2);
    const auto index_ta  = TensorAccessor(index_args, index_addr, K * 4);

    // Logical row 0 of a 32x32 bf16 tile lives in two face rows: face 0 row 0 at
    // byte 0 (cols 0-15) and face 1 row 0 at byte 512 (cols 16-31). Two 32 B
    // reads per tile instead of untilizing the whole tensor first — which would
    // be another op call, and the dispatch floor costs more than this sort.
    cb_reserve_back(SCORES_CB, 1);
    const uint32_t vbuf = get_write_ptr(SCORES_CB);
    for (uint32_t t = 0; t < NT; t++) {
        const uint64_t src = scores_ta.get_noc_addr(t);
        noc_async_read(src, vbuf + t * 64, 32);
        noc_async_read(src + 512, vbuf + t * 64 + 32, 32);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* vals = (volatile tt_l1_ptr uint16_t*)vbuf;

    cb_reserve_back(REC_CB, 1);
    uint32_t* rec  = (uint32_t*)get_write_ptr(REC_CB);
    uint32_t* rec2 = rec + N;
    cb_reserve_back(CNT_CB, 1);
    uint32_t* cnt = (uint32_t*)get_write_ptr(CNT_CB);

    // Map bf16 bits to a key whose UNSIGNED ascending order is descending float
    // order: flip to monotonic-ascending (sign set -> ~v, else v | 0x8000), then
    // invert. record = key << 16 | index, so N must stay within 16 bits.
    for (uint32_t i = 0; i < N; i++) {
        const uint16_t v = vals[i];
        const uint16_t asc =
            (v & 0x8000) ? (uint16_t)~v : (uint16_t)(v | 0x8000);
        const uint16_t key = (uint16_t)(asc ^ 0xFFFF);
        rec[i] = ((uint32_t)key << 16) | i;
    }

    // Two stable counting-sort passes over the key bytes (bits 16-23, then
    // 24-31). After the second pass the records are back in `rec`, fully sorted.
    for (uint32_t shift = 16; shift <= 24; shift += 8) {
        uint32_t* src = (shift == 16) ? rec : rec2;
        uint32_t* dst = (shift == 16) ? rec2 : rec;
        for (uint32_t b = 0; b < 256; b++) {
            cnt[b] = 0;
        }
        for (uint32_t i = 0; i < N; i++) {
            cnt[(src[i] >> shift) & 0xFF]++;
        }
        uint32_t run = 0;
        for (uint32_t b = 0; b < 256; b++) {
            const uint32_t c = cnt[b];
            cnt[b] = run;
            run += c;
        }
        for (uint32_t i = 0; i < N; i++) {
            const uint32_t r = src[i];
            dst[cnt[(r >> shift) & 0xFF]++] = r;
        }
    }

    cb_reserve_back(VAL_CB, 1);
    cb_reserve_back(IDX_CB, 1);
    const uint32_t oval_addr = get_write_ptr(VAL_CB);
    const uint32_t oidx_addr = get_write_ptr(IDX_CB);
    uint16_t* oval = (uint16_t*)oval_addr;
    uint32_t* oidx = (uint32_t*)oidx_addr;
    for (uint32_t j = 0; j < K; j++) {
        const uint32_t r = rec[j];
        const uint32_t i = r & 0xFFFF;
        oidx[j] = i;
        oval[j] = vals[i];  // original bits, not a round-trip through the key
    }
    noc_async_write(oval_addr, values_ta.get_noc_addr(0), K * 2);
    noc_async_write(oidx_addr, index_ta.get_noc_addr(0), K * 4);
    noc_async_write_barrier();
}
