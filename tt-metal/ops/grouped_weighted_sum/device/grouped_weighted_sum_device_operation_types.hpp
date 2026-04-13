// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GroupedWeightedSumParams {
    uint32_t num_groups;
    uint32_t group_dims;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct GroupedWeightedSumInputs {
    const tt::tt_metal::Tensor& features;  // [n, clp, embed_dims]
    const tt::tt_metal::Tensor& weights;   // [n, clp, num_groups]
};

}  // namespace ttnn::prim
