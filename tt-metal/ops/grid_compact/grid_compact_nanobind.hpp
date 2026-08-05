// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::grid_compact {
void bind_grid_compact(nb::module_& mod);
}
