// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "topk_select_nanobind.hpp"
#include <nanobind/nanobind.h>
#include "ttnn-nanobind/bind_function.hpp"
#include "topk_select.hpp"

namespace ttnn::operations::topk_select {

void bind_topk_select(nb::module_& mod) {
    ttnn::bind_function<"topk_select">(
        mod,
        "Top-k over a single row of scores (TILE bf16, one logical row). Writes "
        "the k best scores descending into values ([1,1,1,k] bf16 ROW_MAJOR) and "
        "their source positions into indices ([1,1,1,k] uint32 ROW_MAJOR), ties "
        "resolved to the lower index like torch.topk. Replaces ttnn.topk for "
        "short rows, where tile padding makes the stock bitonic sort do 32x the "
        "needed work on one core.",
        &ttnn::topk_select,
        nb::arg("scores"), nb::arg("values"), nb::arg("indices"), nb::arg("k"));
}

}  // namespace ttnn::operations::topk_select
