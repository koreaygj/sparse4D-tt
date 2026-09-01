// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "anchor_bucket.hpp"
#include "device/anchor_bucket_device_operation.hpp"

namespace ttnn::operations::anchor_bucket {

void anchor_bucket(
    const tt::tt_metal::Tensor& flags, tt::tt_metal::Tensor& perm,
    tt::tt_metal::Tensor& inv, tt::tt_metal::Tensor& live,
    uint32_t num_anchors) {
    ttnn::prim::anchor_bucket(flags, perm, inv, live, num_anchors);
}

}  // namespace ttnn::operations::anchor_bucket
