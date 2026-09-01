// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Paired row gather for the instance bank's top-k selection:
//   out_a[r,:] = src_a[idx[r],:]   (features, bf16)
//   out_b[r,:] = src_b[idx[r],:]   (anchors, fp32)
// Both selections always share one index vector, so one op replaces two
// ttnn.gather calls plus the index-expansion chain each of them needed
// (typecast + reshape + to_layout + repeat_interleave). ttnn.gather ran the
// production shape on 4 cores at 1.1 GB/s — latency-bound, 655 us/call.
struct RowGatherParams {
    uint32_t k;  // rows to gather
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct RowGatherInputs {
    const tt::tt_metal::Tensor& src_a;    // DEVICE TILE [1, N, Ca]
    const tt::tt_metal::Tensor& indices;  // DEVICE ROW_MAJOR uint32 [1,1,1,K]
    const tt::tt_metal::Tensor& out_a;    // DEVICE TILE [1, K, Ca], preallocated
    // Optional second pair riding the same index read (the instance bank's
    // feature/anchor selections always travel together).
    std::optional<tt::tt_metal::Tensor> src_b;
    std::optional<tt::tt_metal::Tensor> out_b;
};

}  // namespace ttnn::prim
