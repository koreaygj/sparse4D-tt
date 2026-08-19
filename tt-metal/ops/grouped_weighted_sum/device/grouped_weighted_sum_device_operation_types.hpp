// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GroupedWeightedSumParams {
    uint32_t num_groups;
    uint32_t group_dims;
    // Dead-pair skip: clp indices map to cameras as cam = clp / clp_per_cam;
    // 0 disables. Requires the perm/live tensors below.
    uint32_t clp_per_cam;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct GroupedWeightedSumInputs {
    const tt::tt_metal::Tensor& features;  // [n, clp, embed_dims]
    const tt::tt_metal::Tensor& weights;   // [n, clp, num_groups]
    // SKIP mode (anchor_bucket outputs): anchors sorted by camera-visibility
    // pattern. perm [1,1,1,>=N] u32 RM maps sorted position -> original anchor;
    // live [1,1,1,32] u32 RM holds per-tile-row live-camera bits. Skipped
    // (row, clp) terms are exactly 0.0 today (masked weights), so the output
    // stays bit-identical.
    std::optional<tt::tt_metal::Tensor> perm;
    std::optional<tt::tt_metal::Tensor> live;
    // L1-sharded u32 scratch, one 32-word shard per core at a single base
    // address: the reader->compute count mailbox (MATH cannot address CBs).
    std::optional<tt::tt_metal::Tensor> mbox;
};

}  // namespace ttnn::prim
