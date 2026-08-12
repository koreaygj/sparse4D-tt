// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::grid_precompute {
void bind_grid_precompute(nb::module_& mod);
}
