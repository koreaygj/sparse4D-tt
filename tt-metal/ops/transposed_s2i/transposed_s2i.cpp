// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "transposed_s2i.hpp"
#include "device/transposed_s2i_device_operation.hpp"

namespace ttnn::operations::transposed_s2i {

void transposed_s2i(
    const tt::tt_metal::Tensor& input, tt::tt_metal::Tensor& output,
    uint32_t num_cams, uint32_t num_pts, uint32_t num_anchors,
    uint32_t num_levels, uint32_t level) {
    ttnn::prim::transposed_s2i(input, output, num_cams, num_pts, num_anchors, num_levels, level);
}

}  // namespace ttnn::operations::transposed_s2i
