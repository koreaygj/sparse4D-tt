// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"
#include "grouped_weighted_sum_device_operation_types.hpp"
#include "grouped_weighted_sum_program_factory.hpp"

namespace ttnn::prim {

struct GroupedWeightedSumOperation {
    using operation_attributes_t = GroupedWeightedSumParams;
    using tensor_args_t = GroupedWeightedSumInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = tt::tt_metal::Tensor;
    using program_factory_t = std::variant<GroupedWeightedSumProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
};

tt::tt_metal::Tensor grouped_weighted_sum(
    const tt::tt_metal::Tensor& features,
    const tt::tt_metal::Tensor& weights,
    uint32_t num_groups, uint32_t group_dims,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::prim
