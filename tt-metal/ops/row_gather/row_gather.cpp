// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "row_gather.hpp"
#include "device/row_gather_device_operation.hpp"

namespace ttnn::operations::row_gather {

void row_gather(
    const tt::tt_metal::Tensor& src_a, const tt::tt_metal::Tensor& indices,
    tt::tt_metal::Tensor& out_a,
    const std::optional<tt::tt_metal::Tensor>& src_b,
    const std::optional<tt::tt_metal::Tensor>& out_b, uint32_t k) {
    ttnn::prim::row_gather(src_a, indices, out_a, src_b, out_b, k);
}

}  // namespace ttnn::operations::row_gather
