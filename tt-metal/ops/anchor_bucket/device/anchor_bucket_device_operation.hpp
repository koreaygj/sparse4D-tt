// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"
#include "anchor_bucket_device_operation_types.hpp"
#include "anchor_bucket_program_factory.hpp"

namespace ttnn::prim {

struct AnchorBucketOperation {
    using operation_attributes_t = AnchorBucketParams;
    using tensor_args_t = AnchorBucketInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = tt::tt_metal::Tensor;
    using program_factory_t = std::variant<AnchorBucketProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
};

void anchor_bucket(
    const tt::tt_metal::Tensor& flags,
    tt::tt_metal::Tensor& perm,
    tt::tt_metal::Tensor& inv,
    tt::tt_metal::Tensor& live,
    uint32_t num_anchors);

}  // namespace ttnn::prim
