// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::topk_select {
void bind_topk_select(nb::module_& mod);
}
