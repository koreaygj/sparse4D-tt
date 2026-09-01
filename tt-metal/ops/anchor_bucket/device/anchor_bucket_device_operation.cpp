// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "anchor_bucket_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

AnchorBucketOperation::program_factory_t AnchorBucketOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return AnchorBucketProgramFactory{};
}

void AnchorBucketOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    TT_FATAL(t.flags.dtype() == DataType::BFLOAT16, "flags must be BFLOAT16");
    TT_FATAL(t.flags.layout() == Layout::ROW_MAJOR, "flags must be ROW_MAJOR");
    TT_FATAL(t.flags.logical_shape()[-1] >= a.num_anchors, "flags row too short");
    TT_FATAL(t.flags.logical_shape()[0] <= 8, "at most 8 cameras (3-bit pattern)");
    for (const Tensor* x : {&t.perm, &t.inv, &t.live}) {
        TT_FATAL(x->dtype() == DataType::UINT32, "outputs must be UINT32");
        TT_FATAL(x->layout() == Layout::ROW_MAJOR, "outputs must be ROW_MAJOR");
    }
    TT_FATAL(t.perm.logical_shape()[-1] >= a.num_anchors, "perm too short");
    TT_FATAL(t.live.logical_shape()[-1] >= 32, "live must hold 32 rows");
}

TensorSpec AnchorBucketOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.perm.tensor_spec();
}

Tensor AnchorBucketOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.perm;
}

void anchor_bucket(
    const Tensor& flags, Tensor& perm, Tensor& inv, Tensor& live,
    uint32_t num_anchors) {
    using Op = AnchorBucketOperation;
    ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_anchors = num_anchors,
            .output_mem_config = perm.memory_config()},
        Op::tensor_args_t{.flags = flags, .perm = perm, .inv = inv, .live = live});
}

}  // namespace ttnn::prim
