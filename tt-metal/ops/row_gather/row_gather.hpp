// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::row_gather {

void row_gather(
    const ttnn::Tensor& src_a, const ttnn::Tensor& indices, ttnn::Tensor& out_a,
    const std::optional<ttnn::Tensor>& src_b,
    const std::optional<ttnn::Tensor>& out_b,
    uint32_t k);

}  // namespace operations::row_gather
using ttnn::operations::row_gather::row_gather;
}  // namespace ttnn
