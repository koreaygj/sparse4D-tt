// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Transposed s2i: L1 sharded [nc, N, K, C] → DRAM [CLP, N, C] in camera-major order
// CLP index = cam * NL * K + level * K + pt  (camera-major, matches concat+transpose)
// Called once per level with level index as compile-time arg.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

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

    constexpr uint32_t TA_OFFSET = 6;
    constexpr auto out_ta = TensorAccessorArgs<TA_OFFSET>();
    const auto out_acc = TensorAccessor(out_ta, out_addr, stick_size);

    // Input stick ordering: cam0_anc0_pt0..pt12, cam0_anc1_pt0..., cam1_anc0_pt0...
    // Global stick i → cam = i / (N*K), anchor = (i % (N*K)) / K, pt = i % K
    // Camera-major CLP: cam * NL * K + LEVEL * K + pt
    // Output page = CLP * N + anchor

    for (uint32_t s = 0; s < num_sticks; s++) {
        uint32_t global_stick = stick_offset + s;
        uint32_t cam = global_stick / (N * K);
        uint32_t rem = global_stick % (N * K);
        uint32_t anchor = rem / K;
        uint32_t pt = rem % K;

        uint32_t clp = cam * NL * K + LEVEL * K + pt;
        uint32_t page_id = clp * N + anchor;
        uint32_t l1_addr = in_l1_base + s * stick_size;

        noc_async_write(l1_addr, out_acc.get_noc_addr(page_id), stick_size);
    }
    noc_async_write_barrier();
}
