// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Top-k over a single row of scores, replacing ttnn.topk for the instance
// bank's two selections (top-300 and top-600 of 900 confidences).
//
// Exists because ttnn.topk tile-pads [1, N] to [32, N32] and bitonic-sorts all
// 32 rows on one core — 97% padding work, 1.9-2.8 ms per call. One radix sort
// over the real row is ~5k integer L1 ops.
struct TopkSelectParams {
    uint32_t n;  // scores in the row (<= 65536: the index shares a 32-bit record)
    uint32_t k;  // entries to emit, descending, ties to the lower index
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct TopkSelectInputs {
    const tt::tt_metal::Tensor& scores;  // DEVICE TILE bf16, volume n in the last dim
    // Preallocated DEVICE ROW_MAJOR outputs, written every call:
    const tt::tt_metal::Tensor& values;   // [1,1,1,k] bf16 — scores, sorted descending
    const tt::tt_metal::Tensor& indices;  // [1,1,1,k] uint32 — source positions
};

}  // namespace ttnn::prim
