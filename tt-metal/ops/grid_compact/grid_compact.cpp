// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "grid_compact.hpp"
#include "device/grid_compact_device_operation.hpp"

namespace ttnn::operations::grid_compact {

void grid_compact(
    const tt::tt_metal::Tensor& grid, tt::tt_metal::Tensor& cgrid,
    tt::tt_metal::Tensor& index, tt::tt_metal::Tensor& flags,
    uint32_t num_pts, uint32_t capacity, uint32_t anchors,
    float threshold_x, float threshold_y,
    const std::optional<tt::tt_metal::Tensor>& bidx) {
    ttnn::prim::grid_compact(
        grid, cgrid, index, flags, num_pts, capacity, anchors, threshold_x, threshold_y, bidx);
}

}  // namespace ttnn::operations::grid_compact
