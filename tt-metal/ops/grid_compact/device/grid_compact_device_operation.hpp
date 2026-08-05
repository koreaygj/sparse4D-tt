// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"
#include "grid_compact_device_operation_types.hpp"
#include "grid_compact_program_factory.hpp"

namespace ttnn::prim {

struct GridCompactOperation {
    using operation_attributes_t = GridCompactParams;
    using tensor_args_t = GridCompactInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = tt::tt_metal::Tensor;
    using program_factory_t = std::variant<GridCompactProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
};

void grid_compact(
    const tt::tt_metal::Tensor& grid,
    tt::tt_metal::Tensor& cgrid,
    tt::tt_metal::Tensor& index,
    tt::tt_metal::Tensor& flags,
    uint32_t num_pts, uint32_t capacity, uint32_t anchors,
    float threshold_x, float threshold_y,
    const std::optional<tt::tt_metal::Tensor>& bidx = std::nullopt);

}  // namespace ttnn::prim
