// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct KpsProjectFusedParams {
    uint32_t num_cams;
    uint32_t num_pts;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool use_tile_compute = false;
    // For precomputed grid output mode:
    bool precompute_grid = false;
    std::vector<std::pair<uint32_t, uint32_t>> spatial_shapes;  // [(H,W)] per FPN level
};

struct KpsProjectFusedInputs {
    const tt::tt_metal::Tensor& key_points;     // [n, num_pts, 3] ROW_MAJOR f32
    const tt::tt_metal::Tensor& anchor;         // [n, 1, 11] ROW_MAJOR f32
    const tt::tt_metal::Tensor& projection_mat; // [nc, 4, 4] ROW_MAJOR f32
    const tt::tt_metal::Tensor& image_wh;       // [nc, 1, 2] ROW_MAJOR f32
};

}  // namespace ttnn::prim
