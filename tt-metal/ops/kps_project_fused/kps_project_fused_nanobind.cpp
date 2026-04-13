// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "kps_project_fused_nanobind.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "kps_project_fused.hpp"

namespace ttnn::operations::kps_project_fused {

void bind_kps_project_fused(nb::module_& mod) {
    const auto* const doc = R"doc(
        Fused KPS rotation + translation + projection + normalize.
        Takes pre-rotation 3D key points and produces normalized 2D grid coordinates.
        All computation in a single dispatch.

        Args:
            key_points: [n, num_pts, 3] ROW_MAJOR float32
            anchor: [n, 1, 11] ROW_MAJOR float32
            projection_mat: [nc, 4, 4] ROW_MAJOR float32
            image_wh: [nc, 1, 2] ROW_MAJOR float32
            num_cams: number of cameras
            num_pts: number of key points per anchor
    )doc";

    ttnn::bind_function<"kps_project_fused">(
        mod, doc, &ttnn::kps_project_fused,
        nb::arg("key_points"), nb::arg("anchor"),
        nb::arg("projection_mat"), nb::arg("image_wh"),
        nb::arg("num_cams"), nb::arg("num_pts"),
        nb::kw_only(), nb::arg("memory_config") = nb::none(),
        nb::arg("use_tile_compute") = false,
        nb::arg("precompute_grid") = false,
        nb::arg("spatial_shapes") = std::vector<std::pair<uint32_t, uint32_t>>{});
}

}  // namespace ttnn::operations::kps_project_fused
