// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/device_operation.hpp"
#include "kps_project_fused_device_operation_types.hpp"
#include "kps_project_fused_program_factory.hpp"
#include "kps_project_fused_tile_program_factory.hpp"

namespace ttnn::prim {

struct KpsProjectFusedOperation {
    using operation_attributes_t = KpsProjectFusedParams;
    using tensor_args_t = KpsProjectFusedInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = tt::tt_metal::Tensor;
    using program_factory_t = std::variant<KpsProjectFusedProgramFactory, KpsProjectFusedTileProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
};

tt::tt_metal::Tensor kps_project_fused(
    const tt::tt_metal::Tensor& key_points,
    const tt::tt_metal::Tensor& anchor,
    const tt::tt_metal::Tensor& projection_mat,
    const tt::tt_metal::Tensor& image_wh,
    uint32_t num_cams, uint32_t num_pts,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    bool use_tile_compute = false,
    bool precompute_grid = false,
    const std::vector<std::pair<uint32_t, uint32_t>>& spatial_shapes = {});

}  // namespace ttnn::prim
