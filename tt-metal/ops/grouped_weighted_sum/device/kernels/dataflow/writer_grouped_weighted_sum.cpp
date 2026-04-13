// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Writer: 2-phase reduction output.
// Output [num_chunks * N, E] — each chunk writes to its section.

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr       = get_arg_val<uint32_t>(0);
    uint32_t num_wus        = get_arg_val<uint32_t>(1);
    uint32_t start_wu       = get_arg_val<uint32_t>(2);
    uint32_t N_TR_TOTAL     = get_arg_val<uint32_t>(3);  // output tile rows per chunk
    uint32_t num_chunks     = get_arg_val<uint32_t>(4);

    constexpr uint32_t out_cb         = get_compile_time_arg_val(0);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t G              = get_compile_time_arg_val(2);
    constexpr uint32_t N_TR           = get_compile_time_arg_val(3);  // same as N_TR_TOTAL

    constexpr auto out_args = TensorAccessorArgs<4>();
    const auto out_acc = TensorAccessor(out_args, out_addr, out_tile_bytes);

    // Output tile layout: [num_chunks * N_padded, E]
    // chunk c, tile_row t, group g → tile_id = (c * N_TR + t) * G + g

    for (uint32_t wu = 0; wu < num_wus; wu++) {
        uint32_t wu_id = start_wu + wu;
        uint32_t n_tr = wu_id % N_TR_TOTAL;
        uint32_t chunk_id = wu_id / N_TR_TOTAL;

        cb_wait_front(out_cb, G);
        uint32_t l1_addr = get_read_ptr(out_cb);
        uint32_t base_tile = (chunk_id * N_TR + n_tr) * G;

        for (uint32_t g = 0; g < G; g++) {
            noc_async_write(l1_addr + g * out_tile_bytes,
                           out_acc.get_noc_addr(base_tile + g), out_tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(out_cb, G);
    }
}
