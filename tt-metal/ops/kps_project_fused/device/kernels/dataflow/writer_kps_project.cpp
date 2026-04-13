// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer: drains output CB to DRAM
// Output layout: [nc, total_anchors, 1, num_pts*2] ROW_MAJOR
// Each CB page = one (cam, anchor) pair = num_pts*2 floats
// Pages are written in order: cam0_anchor0, cam1_anchor0, ..., camN_anchor0, cam0_anchor1, ...

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t out_addr       = get_arg_val<uint32_t>(0);
    uint32_t num_anchors    = get_arg_val<uint32_t>(1);
    uint32_t anchor_offset  = get_arg_val<uint32_t>(2);
    uint32_t total_anchors  = get_arg_val<uint32_t>(3);

    // Compile-time args
    constexpr uint32_t out_cb_index  = get_compile_time_arg_val(0);
    constexpr uint32_t out_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t NC            = get_compile_time_arg_val(2);  // nc (standard) or nc*NL (precompute)
    constexpr uint32_t NL            = get_compile_time_arg_val(3);  // 0=standard, >0=precompute level count
    constexpr uint32_t NC_CAM        = (NL > 0) ? (NC / NL) : NC;   // actual camera count

    constexpr auto out_args = TensorAccessorArgs<4>();
    const auto out_accessor = TensorAccessor(out_args, out_addr, out_page_size);

    for (uint32_t a_idx = 0; a_idx < num_anchors; a_idx++) {
        uint32_t a = anchor_offset + a_idx;
        cb_wait_front(out_cb_index, NC);
        for (uint32_t c = 0; c < NC; c++) {
            uint32_t l1_addr = get_read_ptr(out_cb_index);
            uint32_t page_id;
            if constexpr (NL > 0) {
                // CB order: camera-major [cam0_L0, cam0_L1, ..., cam0_L3, cam1_L0, ...]
                // DRAM order: level-major [L0_C0, L0_C1, L0_C2, L1_C0, ...]
                uint32_t cam = c / NL;
                uint32_t lvl = c % NL;
                page_id = (lvl * NC_CAM + cam) * total_anchors + a;
            } else {
                page_id = c * total_anchors + a;
            }
            uint64_t noc_addr = out_accessor.get_noc_addr(page_id);
            noc_async_write(l1_addr, noc_addr, out_page_size);
            cb_pop_front(out_cb_index, 1);
        }
        noc_async_write_barrier();
    }
}
