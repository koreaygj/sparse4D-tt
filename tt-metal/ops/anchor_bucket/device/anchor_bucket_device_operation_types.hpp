// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Buckets anchors by camera-visibility pattern (from grid_compact's flags)
// so grouped_weighted_sum can skip dead (tile-row, clp) iterations. See the
// kernel header for the perm/inv/live contracts.
struct AnchorBucketParams {
    uint32_t num_anchors;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct AnchorBucketInputs {
    const tt::tt_metal::Tensor& flags;  // [nc,1,1,FW] bf16 RM, 1.0 = live
    const tt::tt_metal::Tensor& perm;   // [1,1,1,NPAD] u32 RM, preallocated
    const tt::tt_metal::Tensor& inv;    // [1,1,1,NPAD] u32 RM, preallocated
    const tt::tt_metal::Tensor& live;   // [1,1,1,32]  u32 RM, preallocated
};

}  // namespace ttnn::prim
