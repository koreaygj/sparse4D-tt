// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "grid_precompute.hpp"
#include "device/grid_precompute_device_operation.hpp"

namespace ttnn::operations::grid_precompute {

void grid_precompute(
    const ttnn::Tensor& cgrid, const ttnn::Tensor& consts,
    ttnn::Tensor& out0, ttnn::Tensor& out1, ttnn::Tensor& out2, ttnn::Tensor& out3,
    uint32_t num_pts) {
    ttnn::prim::grid_precompute(cgrid, consts, out0, out1, out2, out3, num_pts);
}

}  // namespace ttnn::operations::grid_precompute
