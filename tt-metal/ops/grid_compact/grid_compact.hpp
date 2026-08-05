// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::grid_compact {

void grid_compact(
    const ttnn::Tensor& grid, ttnn::Tensor& cgrid, ttnn::Tensor& index, ttnn::Tensor& flags,
    uint32_t num_pts, uint32_t capacity, uint32_t anchors,
    float threshold_x, float threshold_y,
    const std::optional<ttnn::Tensor>& bidx = std::nullopt);

}  // namespace operations::grid_compact
using ttnn::operations::grid_compact::grid_compact;
}  // namespace ttnn
