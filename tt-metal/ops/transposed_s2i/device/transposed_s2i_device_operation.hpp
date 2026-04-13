// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"
#include "transposed_s2i_device_operation_types.hpp"
#include "transposed_s2i_program_factory.hpp"

namespace ttnn::prim {

struct TransposedS2iOperation {
    using operation_attributes_t = TransposedS2iParams;
    using tensor_args_t = TransposedS2iInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = tt::tt_metal::Tensor;
    using program_factory_t = std::variant<TransposedS2iProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
};

void transposed_s2i(
    const tt::tt_metal::Tensor& input,
    tt::tt_metal::Tensor& output,
    uint32_t num_cams, uint32_t num_pts, uint32_t num_anchors,
    uint32_t num_levels, uint32_t level);

}  // namespace ttnn::prim
