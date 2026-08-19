// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"
#include "row_gather_device_operation_types.hpp"
#include "row_gather_program_factory.hpp"

namespace ttnn::prim {

struct RowGatherOperation {
    using operation_attributes_t = RowGatherParams;
    using tensor_args_t = RowGatherInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = tt::tt_metal::Tensor;
    using program_factory_t = std::variant<RowGatherProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
};

void row_gather(
    const tt::tt_metal::Tensor& src_a,
    const tt::tt_metal::Tensor& indices,
    tt::tt_metal::Tensor& out_a,
    const std::optional<tt::tt_metal::Tensor>& src_b,
    const std::optional<tt::tt_metal::Tensor>& out_b,
    uint32_t k);

}  // namespace ttnn::prim
