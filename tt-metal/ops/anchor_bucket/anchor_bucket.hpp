// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations::anchor_bucket {

void anchor_bucket(
    const ttnn::Tensor& flags, ttnn::Tensor& perm, ttnn::Tensor& inv,
    ttnn::Tensor& live, uint32_t num_anchors);

}  // namespace operations::anchor_bucket
using ttnn::operations::anchor_bucket::anchor_bucket;
}  // namespace ttnn
