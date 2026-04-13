// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_weighted_sum.hpp"
#include "device/grouped_weighted_sum_device_operation.hpp"

namespace ttnn::operations::grouped_weighted_sum {

tt::tt_metal::Tensor grouped_weighted_sum(
    const tt::tt_metal::Tensor& features,
    const tt::tt_metal::Tensor& weights,
    uint32_t num_groups,
    uint32_t group_dims,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    return ttnn::prim::grouped_weighted_sum(
        features, weights, num_groups, group_dims, memory_config);
}

}  // namespace ttnn::operations::grouped_weighted_sum
