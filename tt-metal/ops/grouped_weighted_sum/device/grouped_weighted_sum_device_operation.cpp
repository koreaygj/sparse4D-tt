// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_weighted_sum_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

GroupedWeightedSumOperation::program_factory_t GroupedWeightedSumOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GroupedWeightedSumProgramFactory{};
}

void GroupedWeightedSumOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    TT_FATAL(t.features.storage_type() == StorageType::DEVICE, "features must be on device");
    TT_FATAL(t.weights.storage_type() == StorageType::DEVICE, "weights must be on device");
    TT_FATAL(t.features.layout() == Layout::TILE || t.features.layout() == Layout::ROW_MAJOR,
             "features must be TILE or ROW_MAJOR");
    TT_FATAL(t.weights.layout() == Layout::TILE, "weights must be TILE");
    TT_FATAL(t.features.logical_shape().rank() == 3, "features must be 3D [n, clp, E]");
    TT_FATAL(t.weights.logical_shape().rank() == 3, "weights must be 3D [n, clp, G]");
    TT_FATAL(t.features.logical_shape()[0] == t.weights.logical_shape()[0], "n must match");
    TT_FATAL(t.features.logical_shape()[1] == t.weights.logical_shape()[1], "clp must match");
    TT_FATAL(t.features.logical_shape()[-1] == attrs.num_groups * attrs.group_dims, "E must be G*D");
    TT_FATAL(t.weights.logical_shape()[-1] == attrs.num_groups, "last dim must be num_groups");
    TT_FATAL(attrs.group_dims == 32, "group_dims must be 32 (== TILE_WIDTH) in current implementation");
}

TensorSpec GroupedWeightedSumOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    const uint32_t output_n = t.features.logical_shape()[1];  // N (output anchors)
    const uint32_t embed_dims = attrs.num_groups * attrs.group_dims;
    // Output: [num_chunks * output_n_padded, embed_dims] TILE — 2-phase reduction
    const uint32_t num_chunks = 2;
    const uint32_t output_n_padded = ((output_n + 31) / 32) * 32;
    const ttnn::Shape output_shape({num_chunks * output_n_padded, embed_dims});
    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT16,
            PageConfig(Layout::TILE),
            attrs.output_mem_config,
            output_shape, output_shape));
}

Tensor GroupedWeightedSumOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.features.device());
}

Tensor grouped_weighted_sum(
    const Tensor& features, const Tensor& weights,
    uint32_t num_groups, uint32_t group_dims,
    const std::optional<MemoryConfig>& memory_config) {
    using Op = GroupedWeightedSumOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_groups = num_groups, .group_dims = group_dims,
            .output_mem_config = memory_config.value_or(features.memory_config()),
        },
        Op::tensor_args_t{.features = features, .weights = weights});
}

}  // namespace ttnn::prim
