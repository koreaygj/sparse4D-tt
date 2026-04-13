// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "transposed_s2i_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

// TransposedS2i: in-place write to pre-allocated output
TransposedS2iOperation::program_factory_t TransposedS2iOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return TransposedS2iProgramFactory{};
}

void TransposedS2iOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& t) {
    TT_FATAL(t.input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
             "input must be L1 HEIGHT_SHARDED");
    TT_FATAL(t.output.storage_type() == StorageType::DEVICE, "output must be on device");
}

TensorSpec TransposedS2iOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& t) {
    return t.output.tensor_spec();
}

Tensor TransposedS2iOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.output;
}

void transposed_s2i(
    const Tensor& input, Tensor& output,
    uint32_t num_cams, uint32_t num_pts, uint32_t num_anchors,
    uint32_t num_levels, uint32_t level) {
    using Op = TransposedS2iOperation;
    ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_cams = num_cams, .num_pts = num_pts,
            .num_anchors = num_anchors, .num_levels = num_levels,
            .level = level,
            .output_mem_config = output.memory_config(),
        },
        Op::tensor_args_t{.input = input, .output = output});
}

}  // namespace ttnn::prim
