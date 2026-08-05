// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::transposed_s2i {

void transposed_s2i(
    const ttnn::Tensor& input, ttnn::Tensor& output,
    uint32_t num_cams, uint32_t num_pts, uint32_t num_anchors,
    uint32_t num_levels, uint32_t level,
    const std::optional<ttnn::Tensor>& index = std::nullopt,
    uint32_t capacity = 0);

}  // namespace operations::transposed_s2i
using ttnn::operations::transposed_s2i::transposed_s2i;
}  // namespace ttnn
