// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::grouped_weighted_sum {

ttnn::Tensor grouped_weighted_sum(
    const ttnn::Tensor& features,   // [n, clp, embed_dims] ROW_MAJOR bf16
    const ttnn::Tensor& weights,    // [n, clp, num_groups] ROW_MAJOR bf16
    uint32_t num_groups,
    uint32_t group_dims,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace operations::grouped_weighted_sum
using ttnn::operations::grouped_weighted_sum::grouped_weighted_sum;
}  // namespace ttnn
