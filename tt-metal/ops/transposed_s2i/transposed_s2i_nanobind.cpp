// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "transposed_s2i_nanobind.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "transposed_s2i.hpp"

namespace ttnn::operations::transposed_s2i {

void bind_transposed_s2i(nb::module_& mod) {
    ttnn::bind_function<"transposed_s2i">(
        mod,
        "Transposed sharded-to-interleaved: writes L1 sharded [nc,N,K,C] to DRAM [CLP,N,C] in camera-major CLP order.",
        &ttnn::transposed_s2i,
        nb::arg("input"), nb::arg("output"),
        nb::arg("num_cams"), nb::arg("num_pts"), nb::arg("num_anchors"),
        nb::arg("num_levels"), nb::arg("level"));
}

}  // namespace ttnn::operations::transposed_s2i
