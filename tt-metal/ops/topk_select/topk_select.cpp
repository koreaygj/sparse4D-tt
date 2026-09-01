// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "topk_select.hpp"
#include "device/topk_select_device_operation.hpp"

namespace ttnn::operations::topk_select {

void topk_select(
    const tt::tt_metal::Tensor& scores,
    tt::tt_metal::Tensor& values,
    tt::tt_metal::Tensor& indices,
    uint32_t k) {
    ttnn::prim::topk_select(scores, values, indices, k);
}

}  // namespace ttnn::operations::topk_select
