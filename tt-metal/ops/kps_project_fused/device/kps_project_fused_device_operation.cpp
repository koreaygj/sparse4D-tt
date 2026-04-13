// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "kps_project_fused_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

KpsProjectFusedOperation::program_factory_t KpsProjectFusedOperation::select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    if (attrs.use_tile_compute) {
        return KpsProjectFusedTileProgramFactory{};
    }
    return KpsProjectFusedProgramFactory{};
}

void KpsProjectFusedOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    TT_FATAL(t.key_points.storage_type() == StorageType::DEVICE, "key_points must be on device");
    TT_FATAL(t.key_points.layout() == Layout::ROW_MAJOR, "key_points must be ROW_MAJOR");
    TT_FATAL(t.key_points.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "INTERLEAVED only");
    if (attrs.precompute_grid && t.key_points.logical_shape().rank() == 4) {
        // Standalone precompute mode: input is [nc, N, 1, pts*2] f32 normalized grid
        TT_FATAL(!attrs.spatial_shapes.empty(), "spatial_shapes required for precompute_grid");
    } else if (attrs.precompute_grid && t.key_points.logical_shape().rank() == 3) {
        // Fused precompute mode: full kps_project + precompute in one pass
        TT_FATAL(!attrs.spatial_shapes.empty(), "spatial_shapes required for precompute_grid");
        TT_FATAL(t.anchor.storage_type() == StorageType::DEVICE, "anchor must be on device");
        TT_FATAL(t.projection_mat.storage_type() == StorageType::DEVICE, "projection_mat must be on device");
        TT_FATAL(t.image_wh.storage_type() == StorageType::DEVICE, "image_wh must be on device");
        TT_FATAL(t.key_points.logical_shape()[-1] == 3, "key_points last dim must be 3");
    } else {
        TT_FATAL(t.anchor.storage_type() == StorageType::DEVICE, "anchor must be on device");
        TT_FATAL(t.projection_mat.storage_type() == StorageType::DEVICE, "projection_mat must be on device");
        TT_FATAL(t.image_wh.storage_type() == StorageType::DEVICE, "image_wh must be on device");
        TT_FATAL(t.key_points.logical_shape().rank() == 3, "key_points must be 3D [n, num_pts, 3]");
        TT_FATAL(t.key_points.logical_shape()[-1] == 3, "key_points last dim must be 3");
    }
}

TensorSpec KpsProjectFusedOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (attrs.precompute_grid && t.key_points.logical_shape().rank() == 4) {
        // Standalone precompute: input [nc, N, 1, pts*2] → n is dim[1]
        const uint32_t n = t.key_points.logical_shape()[1];
        const uint32_t NL = attrs.spatial_shapes.size();
        const ttnn::Shape output_shape({attrs.num_cams * NL, n, 1, attrs.num_pts * 6});
        return TensorSpec(
            output_shape,
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT16,
                PageConfig(Layout::ROW_MAJOR),
                attrs.output_mem_config,
                output_shape, output_shape));
    }
    if (attrs.precompute_grid && t.key_points.logical_shape().rank() == 3) {
        // Fused precompute: multi-level output [nc*NL, N, 1, pts*6] bf16
        const uint32_t n = t.key_points.logical_shape()[0];
        const uint32_t NL = attrs.spatial_shapes.size();
        const ttnn::Shape output_shape({attrs.num_cams * NL, n, 1, attrs.num_pts * 6});
        return TensorSpec(
            output_shape,
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT16,
                PageConfig(Layout::ROW_MAJOR),
                attrs.output_mem_config,
                output_shape, output_shape));
    }
    // Standard: input [n, pts, 3] → n is dim[0]
    const uint32_t n = t.key_points.logical_shape()[0];
    const ttnn::Shape output_shape({attrs.num_cams, n, 1, attrs.num_pts * 2});
    return TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            DataType::FLOAT32,
            PageConfig(Layout::ROW_MAJOR),
            attrs.output_mem_config,
            output_shape, output_shape));
}

Tensor KpsProjectFusedOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    return create_device_tensor(compute_output_specs(attrs, t), t.key_points.device());
}

Tensor kps_project_fused(
    const Tensor& key_points, const Tensor& anchor,
    const Tensor& projection_mat, const Tensor& image_wh,
    uint32_t num_cams, uint32_t num_pts,
    const std::optional<MemoryConfig>& memory_config,
    bool use_tile_compute,
    bool precompute_grid,
    const std::vector<std::pair<uint32_t, uint32_t>>& spatial_shapes) {
    using Op = KpsProjectFusedOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_cams = num_cams, .num_pts = num_pts,
            .output_mem_config = memory_config.value_or(key_points.memory_config()),
            .use_tile_compute = use_tile_compute,
            .precompute_grid = precompute_grid,
            .spatial_shapes = spatial_shapes,
        },
        Op::tensor_args_t{
            .key_points = key_points, .anchor = anchor,
            .projection_mat = projection_mat, .image_wh = image_wh,
        });
}

}  // namespace ttnn::prim
