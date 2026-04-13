// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_weighted_sum_nanobind.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "grouped_weighted_sum.hpp"

namespace ttnn::operations::grouped_weighted_sum {

void bind_grouped_weighted_sum(nb::module_& mod) {
    const auto* const doc = R"doc(
        Grouped weighted sum: fused repeat_interleave + multiply + sum.
        Computes output[n,d] = sum_clp(weights[n,clp,g(d)] * features[n,clp,d])
        where g(d) = d // group_dims. No intermediate tensor expansion.

        Args:
            features: [n, clp, embed_dims] ROW_MAJOR bf16
            weights: [n, clp, num_groups] ROW_MAJOR bf16
            num_groups: number of attention groups (e.g. 8)
            group_dims: dims per group (e.g. 32, so embed_dims = num_groups * group_dims)
    )doc";

    ttnn::bind_function<"grouped_weighted_sum">(
        mod, doc, &ttnn::grouped_weighted_sum,
        nb::arg("features"), nb::arg("weights"),
        nb::arg("num_groups"), nb::arg("group_dims"),
        nb::kw_only(), nb::arg("memory_config") = nb::none());
}

}  // namespace ttnn::operations::grouped_weighted_sum
