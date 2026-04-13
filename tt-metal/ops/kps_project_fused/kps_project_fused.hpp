// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::kps_project_fused {

ttnn::Tensor kps_project_fused(
    const ttnn::Tensor& key_points,
    const ttnn::Tensor& anchor,
    const ttnn::Tensor& projection_mat,
    const ttnn::Tensor& image_wh,
    uint32_t num_cams,
    uint32_t num_pts,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool use_tile_compute = false,
    bool precompute_grid = false,
    const std::vector<std::pair<uint32_t, uint32_t>>& spatial_shapes = {});

}  // namespace operations::kps_project_fused
using ttnn::operations::kps_project_fused::kps_project_fused;
}  // namespace ttnn
