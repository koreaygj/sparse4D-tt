// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "row_gather_nanobind.hpp"
#include <nanobind/stl/optional.h>
#include <nanobind/nanobind.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "row_gather.hpp"

namespace ttnn::operations::row_gather {

void bind_row_gather(nb::module_& mod) {
    ttnn::bind_function<"row_gather">(
        mod,
        "Paired row gather: out_a[r] = src_a[idx[r]], out_b[r] = src_b[idx[r]] "
        "for r < k, with idx a uint32 ROW_MAJOR [1,1,1,>=k] vector (e.g. "
        "topk_select's output, no expansion chain needed). src/out are TILE "
        "tensors; both pairs ride one op and the rows fan out across cores "
        "with async NOC pipelining. Bit-identical row copies.",
        &ttnn::row_gather,
        nb::arg("src_a"), nb::arg("indices"), nb::arg("out_a"),
        nb::arg("src_b") = nb::none(), nb::arg("out_b") = nb::none(),
        nb::arg("k"));
}

}  // namespace ttnn::operations::row_gather
