// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Buckets anchors by camera-visibility pattern for grouped_weighted_sum's
// dead-pair skip. Reads grid_compact's flags (1.0/0.0 per (camera, anchor)),
// stable-counting-sorts the anchors by their 3-bit pattern, and emits:
//   perm  [PAD] u32 : sorted position -> original anchor
//   inv   [PAD] u32 : original anchor -> sorted position (output un-permute)
//   live  [32]  u32 : per sorted tile-row, OR of member patterns (bit c = cam
//                     c has at least one live anchor in the row)
// Sorting makes each 32-anchor tile row pattern-homogeneous, which is what
// lets gws skip whole (row, clp) iterations. Integer-only, one core — the
// same soft-float-free discipline as topk_select.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    constexpr uint32_t N      = get_compile_time_arg_val(0);   // anchors (900)
    constexpr uint32_t NC     = get_compile_time_arg_val(1);   // cameras (3)
    constexpr uint32_t FW     = get_compile_time_arg_val(2);   // flags row width
    constexpr uint32_t NPAD   = get_compile_time_arg_val(3);   // perm/inv length
    constexpr uint32_t CB_FLG = get_compile_time_arg_val(4);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(5);
    constexpr uint32_t FPAGE  = get_compile_time_arg_val(6);  // aligned stick bytes

    constexpr auto f_args = TensorAccessorArgs<7>();
    constexpr auto p_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto l_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();

    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_arg_val<uint32_t>(1);
    const uint32_t i_addr = get_arg_val<uint32_t>(2);
    const uint32_t l_addr = get_arg_val<uint32_t>(3);

    const auto fa = TensorAccessor(f_args, f_addr, FPAGE);
    const auto pa = TensorAccessor(p_args, p_addr, NPAD * 4);
    const auto ia = TensorAccessor(i_args, i_addr, NPAD * 4);
    const auto la = TensorAccessor(l_args, l_addr, 32 * 4);

    // NOC DRAM->L1 reads need a 32B-aligned L1 destination; FW*2=1800 is not,
    // so pad the per-camera stride (misalignment silently lands data shifted).
    constexpr uint32_t FSTRIDE = (FW * 2 + 31) & ~31u;  // bytes, 32B-aligned
    cb_reserve_back(CB_FLG, 1);
    const uint32_t fbuf = get_write_ptr(CB_FLG);
    for (uint32_t c = 0; c < NC; c++) {
        noc_async_read(fa.get_noc_addr(c), fbuf + c * FSTRIDE, FW * 2);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* flg = (volatile tt_l1_ptr uint16_t*)fbuf;

    cb_reserve_back(CB_OUT, 1);
    const uint32_t obuf = get_write_ptr(CB_OUT);
    uint32_t* perm = (uint32_t*)obuf;
    uint32_t* inv  = perm + NPAD;
    uint32_t* live = inv + NPAD;
    uint8_t*  pat  = (uint8_t*)(live + 32);

    // Pattern pass: two bf16 flags per uint32 load (the volatile uint16 loop
    // was the op's hot spot at 2700 L1 reads).
    for (uint32_t a = 0; a < N; a++) {
        pat[a] = 0;
    }
    for (uint32_t c = 0; c < NC; c++) {
        const uint8_t bit = (uint8_t)(1u << c);
        volatile tt_l1_ptr uint32_t* f32 =
            (volatile tt_l1_ptr uint32_t*)(fbuf + c * FSTRIDE);
        for (uint32_t a = 0; a < N; a += 2) {
            const uint32_t v = f32[a >> 1];
            if (v & 0xFFFFu) {
                pat[a] |= bit;
            }
            if ((v >> 16) && (a + 1 < N)) {
                pat[a + 1] |= bit;
            }
        }
    }
    uint32_t cnt[8] = {0};
    for (uint32_t a = 0; a < N; a++) {
        cnt[pat[a]]++;
    }
    uint32_t run = 0;
    for (uint32_t b = 0; b < 8; b++) {
        const uint32_t c = cnt[b];
        cnt[b] = run;
        run += c;
    }
    for (uint32_t a = 0; a < N; a++) {          // stable scatter
        const uint32_t pos = cnt[pat[a]]++;
        perm[pos] = a;
        inv[a] = pos;
    }
    for (uint32_t a = N; a < NPAD; a++) {
        perm[a] = 0;
        inv[a] = 0;
    }
    for (uint32_t rt = 0; rt < 32; rt++) {
        uint32_t bits = 0;
        for (uint32_t r = 0; r < 32; r++) {
            const uint32_t sp = rt * 32 + r;
            if (sp < N) {
                bits |= pat[perm[sp]];
            }
        }
        live[rt] = bits;
    }

    noc_async_write((uint32_t)(uintptr_t)perm, pa.get_noc_addr(0), NPAD * 4);
    noc_async_write((uint32_t)(uintptr_t)inv,  ia.get_noc_addr(0), NPAD * 4);
    noc_async_write((uint32_t)(uintptr_t)live, la.get_noc_addr(0), 32 * 4);
    noc_async_write_barrier();
}
