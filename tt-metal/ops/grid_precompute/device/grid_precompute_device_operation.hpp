// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"
#include "grid_precompute_device_operation_types.hpp"
#include "grid_precompute_program_factory.hpp"

namespace ttnn::prim {

struct GridPrecomputeOperation {
    using operation_attributes_t = GridPrecomputeParams;
    using tensor_args_t = GridPrecomputeInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = tt::tt_metal::Tensor;
    using program_factory_t = std::variant<GridPrecomputeProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
};

void grid_precompute(
    const tt::tt_metal::Tensor& cgrid,
    const tt::tt_metal::Tensor& consts,
    tt::tt_metal::Tensor& out0,
    tt::tt_metal::Tensor& out1,
    tt::tt_metal::Tensor& out2,
    tt::tt_metal::Tensor& out3,
    uint32_t num_pts);

}  // namespace ttnn::prim
