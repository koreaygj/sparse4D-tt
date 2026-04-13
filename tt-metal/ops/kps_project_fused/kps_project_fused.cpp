// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "kps_project_fused.hpp"
#include "device/kps_project_fused_device_operation.hpp"

namespace ttnn::operations::kps_project_fused {

tt::tt_metal::Tensor kps_project_fused(
    const tt::tt_metal::Tensor& key_points,
    const tt::tt_metal::Tensor& anchor,
    const tt::tt_metal::Tensor& projection_mat,
    const tt::tt_metal::Tensor& image_wh,
    uint32_t num_cams,
    uint32_t num_pts,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    bool use_tile_compute,
    bool precompute_grid,
    const std::vector<std::pair<uint32_t, uint32_t>>& spatial_shapes) {
    return ttnn::prim::kps_project_fused(
        key_points, anchor, projection_mat, image_wh,
        num_cams, num_pts, memory_config, use_tile_compute,
        precompute_grid, spatial_shapes);
}

}  // namespace ttnn::operations::kps_project_fused
