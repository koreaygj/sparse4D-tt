// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "grid_precompute_nanobind.hpp"
#include <nanobind/nanobind.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "grid_precompute.hpp"

namespace ttnn::operations::grid_precompute {

void bind_grid_precompute(nb::module_& mod) {
    ttnn::bind_function<"grid_precompute">(
        mod,
        "Converts a compacted Q14 sampling grid into grid_sample's precomputed 6-field "
        "form (h0, w0, and the four boundary-masked bilinear weights), one BFLOAT16 "
        "ROW_MAJOR output per FPN level, computed entirely on the Tensix engines. Row r "
        "of every output corresponds 1:1 to cgrid row r, so the index/flags/bidx tensors "
        "from grid_compact apply unchanged. `consts` is the tile pack built by the model "
        "(per-level affine and bound tiles plus the selector matrices); its ordering is "
        "the contract documented in the reader kernel. Exists because deriving these six "
        "values in grid_sample's reader costs ~62% of that op in soft float on an "
        "FPU-less core; feeding the precomputed form measured 2.78x.",
        &ttnn::grid_precompute,
        nb::arg("cgrid"), nb::arg("consts"),
        nb::arg("out0"), nb::arg("out1"), nb::arg("out2"), nb::arg("out3"),
        nb::arg("num_pts"));
}

}  // namespace ttnn::operations::grid_precompute
