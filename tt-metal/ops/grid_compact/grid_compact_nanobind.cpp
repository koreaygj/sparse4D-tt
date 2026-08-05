// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "grid_compact_nanobind.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "grid_compact.hpp"

namespace ttnn::operations::grid_compact {

void bind_grid_compact(nb::module_& mod) {
    ttnn::bind_function<"grid_compact">(
        mod,
        "Compacts a DFA sampling grid to the rows with at least one in-bounds point. "
        "Writes the kept rows to cgrid and their source row ids to index (SENTINEL "
        "0xFFFFFFFF past the kept count); the camera id is stored at slot 2*num_pts. "
        "flags [nc, 1, 1, >=anchors] bf16 gets 1.0 for every kept (camera, anchor) and "
        "0.0 otherwise, so the attention weights of dropped rows can be zeroed. Passing "
        "bidx switches to POOLED mode: one shared list of kept rows instead of a fixed "
        "block per camera, with each row's camera written to bidx for grid_sample's "
        "batch_index. Pooling needs far less capacity because the cameras do not peak "
        "together (measured 902 rows pooled versus 3 x 563 per camera).",
        &ttnn::grid_compact,
        nb::arg("grid"), nb::arg("cgrid"), nb::arg("index"), nb::arg("flags"),
        nb::arg("num_pts"), nb::arg("capacity"), nb::arg("anchors"),
        nb::arg("threshold_x"), nb::arg("threshold_y"),
        nb::arg("bidx") = nb::none());
}

}  // namespace ttnn::operations::grid_compact
