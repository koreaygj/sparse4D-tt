// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "topk_select_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

TopkSelectOperation::program_factory_t TopkSelectOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return TopkSelectProgramFactory{};
}

void TopkSelectOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    TT_FATAL(t.scores.dtype() == DataType::BFLOAT16, "scores must be BFLOAT16");
    TT_FATAL(t.scores.layout() == Layout::TILE, "scores must be TILE");
    TT_FATAL(t.scores.storage_type() == StorageType::DEVICE, "scores must be on device");
    // One real row: the kernel reads logical row 0 only, so any leading dims
    // must be size 1 or the extra rows would be silently ignored.
    TT_FATAL(
        t.scores.logical_shape().volume() == a.n,
        "scores must hold a single row of {} values (volume {})",
        a.n, t.scores.logical_shape().volume());
    TT_FATAL(a.n <= 65536, "index shares a 32-bit record with the key: n <= 65536");
    TT_FATAL(a.k <= a.n, "k ({}) must not exceed n ({})", a.k, a.n);
    TT_FATAL(t.values.dtype() == DataType::BFLOAT16, "values must be BFLOAT16");
    TT_FATAL(t.values.layout() == Layout::ROW_MAJOR, "values must be ROW_MAJOR");
    TT_FATAL(t.values.storage_type() == StorageType::DEVICE, "values must be on device");
    TT_FATAL(
        t.values.logical_shape()[-1] == a.k, "values last dim must equal k ({})", a.k);
    TT_FATAL(t.indices.dtype() == DataType::UINT32, "indices must be UINT32");
    TT_FATAL(t.indices.layout() == Layout::ROW_MAJOR, "indices must be ROW_MAJOR");
    TT_FATAL(t.indices.storage_type() == StorageType::DEVICE, "indices must be on device");
    TT_FATAL(
        t.indices.logical_shape()[-1] == a.k, "indices last dim must equal k ({})", a.k);
}

TensorSpec TopkSelectOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.indices.tensor_spec();
}

Tensor TopkSelectOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.indices;
}

void topk_select(const Tensor& scores, Tensor& values, Tensor& indices, uint32_t k) {
    const uint32_t n = scores.logical_shape()[-1];
    using Op = TopkSelectOperation;
    ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .n = n, .k = k, .output_mem_config = indices.memory_config()},
        Op::tensor_args_t{.scores = scores, .values = values, .indices = indices});
}

}  // namespace ttnn::prim
