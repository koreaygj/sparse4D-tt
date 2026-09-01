// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::row_gather {
void bind_row_gather(nb::module_& mod);
}
