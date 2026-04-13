// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Transposed s2i: writes L1 sharded grid_sample output to DRAM in transposed CLP order
struct TransposedS2iParams {
    uint32_t num_cams;
    uint32_t num_pts;       // K
    uint32_t num_anchors;   // N
    uint32_t num_levels;    // NL
    uint32_t level;         // current level index
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct TransposedS2iInputs {
    const tt::tt_metal::Tensor& input;   // L1 HEIGHT_SHARDED [nc, N, K, C]
    const tt::tt_metal::Tensor& output;  // pre-allocated DRAM [CLP, N, C]
};

}  // namespace ttnn::prim
