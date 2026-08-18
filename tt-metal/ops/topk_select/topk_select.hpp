// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::topk_select {

void topk_select(
    const ttnn::Tensor& scores, ttnn::Tensor& values, ttnn::Tensor& indices, uint32_t k);

}  // namespace operations::topk_select
using ttnn::operations::topk_select::topk_select;
}  // namespace ttnn
