// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace ttnn::operations::anchor_bucket {
void bind_anchor_bucket(nb::module_& mod);
}
