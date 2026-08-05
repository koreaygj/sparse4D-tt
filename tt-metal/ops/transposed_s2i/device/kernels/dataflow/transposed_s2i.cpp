// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Transposed s2i: L1 sharded [nc, N, K, C] → DRAM [CLP, N, C] in camera-major order
// CLP index = cam * NL * K + level * K + pt  (camera-major, matches concat+transpose)
// Called once per level with level index as compile-time arg.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// The index accessor's compile-time args only EXIST when index-driven mode is on, so
// this has to be a template: `if constexpr` inside kernel_main() would not discard the
// branch (kernel_main is not a template) and TensorAccessorArgs<OFF> would still be
// instantiated past the end of the arg list.
template <bool USE_INDEX, uint32_t OFF, uint32_t IDX_CB, uint32_t IDX_COUNT>
FORCE_INLINE volatile tt_l1_ptr uint32_t* load_index(uint32_t idx_addr) {
    if constexpr (USE_INDEX) {
        constexpr auto idx_ta = TensorAccessorArgs<OFF>();
        const auto idx_acc = TensorAccessor(idx_ta, idx_addr, IDX_COUNT * 4);
        const uint32_t idx_l1 = get_write_ptr(IDX_CB);
        noc_async_read(idx_acc.get_noc_addr(0), idx_l1, IDX_COUNT * 4);
        noc_async_read_barrier();
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    } else {
        return nullptr;
    }
}

void kernel_main() {
    uint32_t out_addr       = get_arg_val<uint32_t>(0);
    uint32_t num_sticks     = get_arg_val<uint32_t>(1);
    uint32_t stick_offset   = get_arg_val<uint32_t>(2);
    uint32_t in_l1_base     = get_arg_val<uint32_t>(3);

    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t N          = get_compile_time_arg_val(1);  // 900
    constexpr uint32_t K          = get_compile_time_arg_val(2);  // 13
    constexpr uint32_t NC         = get_compile_time_arg_val(3);  // 3
    constexpr uint32_t NL         = get_compile_time_arg_val(4);  // 4
    constexpr uint32_t LEVEL      = get_compile_time_arg_val(5);  // 0-3
    // Index-driven mode: the input rows are a COMPACTED grid, so a stick's
    // (camera, anchor) can no longer be derived from its position — it comes from
    // idx[]. SENTINEL rows are capacity padding and must not be scattered, or they
    // would overwrite live anchors with stale samples.
    constexpr uint32_t USE_INDEX  = get_compile_time_arg_val(6);
    constexpr uint32_t IDX_CB     = get_compile_time_arg_val(7);
    constexpr uint32_t IDX_COUNT  = get_compile_time_arg_val(8);
    constexpr uint32_t SENTINEL   = 0xFFFFFFFFu;

    constexpr uint32_t TA_OFFSET = 9;
    constexpr auto out_ta = TensorAccessorArgs<TA_OFFSET>();
    const auto out_acc = TensorAccessor(out_ta, out_addr, stick_size);

    volatile tt_l1_ptr uint32_t* idx =
        load_index<USE_INDEX != 0, out_ta.next_compile_time_args_offset(), IDX_CB, IDX_COUNT>(
            get_arg_val<uint32_t>(4));

    // Input stick ordering: cam0_anc0_pt0..pt12, cam0_anc1_pt0..., cam1_anc0_pt0...
    // Global stick i → cam = i / (N*K), anchor = (i % (N*K)) / K, pt = i % K
    // Camera-major CLP: cam * NL * K + LEVEL * K + pt
    // Output page = CLP * N + anchor

    for (uint32_t s = 0; s < num_sticks; s++) {
        uint32_t global_stick = stick_offset + s;
        uint32_t cam, anchor, pt;
        if constexpr (USE_INDEX) {
            pt = global_stick % K;
            const uint32_t src_row = idx[global_stick / K];
            if (src_row == SENTINEL) {
                continue;  // capacity padding: no anchor owns this sample
            }
            cam = src_row / N;
            anchor = src_row % N;
        } else {
            cam = global_stick / (N * K);
            const uint32_t rem = global_stick % (N * K);
            anchor = rem / K;
            pt = rem % K;
        }

        uint32_t clp = cam * NL * K + LEVEL * K + pt;
        uint32_t page_id = clp * N + anchor;
        uint32_t l1_addr = in_l1_base + s * stick_size;

        noc_async_write(l1_addr, out_acc.get_noc_addr(page_id), stick_size);
    }
    noc_async_write_barrier();
}
