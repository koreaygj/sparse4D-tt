// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::grid_precompute {

void grid_precompute(
    const ttnn::Tensor& cgrid,
    const ttnn::Tensor& consts,
    ttnn::Tensor& out0,
    ttnn::Tensor& out1,
    ttnn::Tensor& out2,
    ttnn::Tensor& out3,
    uint32_t num_pts);

}  // namespace operations::grid_precompute
using ttnn::operations::grid_precompute::grid_precompute;
}  // namespace ttnn
