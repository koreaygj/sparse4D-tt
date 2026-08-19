// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "row_gather_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

RowGatherOperation::program_factory_t RowGatherOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RowGatherProgramFactory{};
}

void RowGatherOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    TT_FATAL(t.src_a.layout() == t.out_a.layout(), "a pair layout mismatch");
    for (const Tensor* x : {&t.src_a, &t.out_a}) {
        TT_FATAL(x->storage_type() == StorageType::DEVICE, "tensors must be on device");
    }
    TT_FATAL(t.src_b.has_value() == t.out_b.has_value(), "src_b needs out_b");
    if (t.src_b.has_value()) {
        for (const Tensor* x : {&*t.src_b, &*t.out_b}) {
            TT_FATAL(x->layout() == Layout::TILE, "tensors must be TILE");
            TT_FATAL(x->storage_type() == StorageType::DEVICE, "tensors must be on device");
        }
    }
    TT_FATAL(t.indices.dtype() == DataType::UINT32, "indices must be UINT32");
    TT_FATAL(t.indices.layout() == Layout::ROW_MAJOR, "indices must be ROW_MAJOR");
    TT_FATAL(t.indices.logical_shape()[-1] >= a.k, "indices shorter than k");
    TT_FATAL(t.src_a.dtype() == t.out_a.dtype(), "a dtype mismatch");
    if (t.src_b.has_value()) {
        TT_FATAL(t.src_b->dtype() == t.out_b->dtype(), "b dtype mismatch");
    }
    TT_FATAL(
        t.src_a.padded_shape()[-1] == t.out_a.padded_shape()[-1],
        "a width mismatch");
    if (t.src_b.has_value()) {
        TT_FATAL(
            t.src_b->padded_shape()[-1] == t.out_b->padded_shape()[-1],
            "b width mismatch");
    }
    TT_FATAL(t.out_a.padded_shape()[-2] >= a.k, "out_a shorter than k");
    if (t.out_b.has_value()) {
        TT_FATAL(t.out_b->padded_shape()[-2] >= a.k, "out_b shorter than k");
    }
}

TensorSpec RowGatherOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.out_a.tensor_spec();
}

Tensor RowGatherOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.out_a;
}

void row_gather(
    const Tensor& src_a, const Tensor& indices, Tensor& out_a,
    const std::optional<Tensor>& src_b, const std::optional<Tensor>& out_b,
    uint32_t k) {
    using Op = RowGatherOperation;
    ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .k = k, .output_mem_config = out_a.memory_config()},
        Op::tensor_args_t{
            .src_a = src_a, .indices = indices, .out_a = out_a,
            .src_b = src_b, .out_b = out_b});
}

}  // namespace ttnn::prim
