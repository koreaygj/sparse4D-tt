// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "grid_precompute_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

GridPrecomputeOperation::program_factory_t GridPrecomputeOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GridPrecomputeProgramFactory{};
}

void GridPrecomputeOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    TT_FATAL(t.cgrid.dtype() == DataType::UINT16, "cgrid must be UINT16 (Q14 fixed point)");
    TT_FATAL(t.cgrid.layout() == Layout::ROW_MAJOR, "cgrid must be ROW_MAJOR");
    TT_FATAL(t.cgrid.storage_type() == StorageType::DEVICE, "cgrid must be on device");
    // The constant pack is read tile-by-tile via the accessor, which is only
    // valid when a page IS a tile — i.e. FLOAT32 TILE layout.
    TT_FATAL(t.consts.dtype() == DataType::FLOAT32, "consts must be FLOAT32");
    TT_FATAL(t.consts.layout() == Layout::TILE, "consts must be TILE layout");
    TT_FATAL(t.consts.storage_type() == StorageType::DEVICE, "consts must be on device");
    const uint32_t expected_tiles = 3 * a.num_levels + 5 * ((a.num_pts * 6 + 31) / 32);
    TT_FATAL(
        t.consts.physical_volume() / (32 * 32) >= expected_tiles,
        "consts holds {} tiles, ordering needs {}",
        t.consts.physical_volume() / (32 * 32), expected_tiles);
    for (const Tensor* out : {&t.out0, &t.out1, &t.out2, &t.out3}) {
        TT_FATAL(out->dtype() == DataType::BFLOAT16, "outputs must be BFLOAT16");
        TT_FATAL(out->layout() == Layout::ROW_MAJOR, "outputs must be ROW_MAJOR");
        TT_FATAL(out->storage_type() == StorageType::DEVICE, "outputs must be on device");
        TT_FATAL(
            out->logical_shape()[-1] == a.num_pts * 6,
            "output last dim must be num_pts*6 ({}), got {}",
            a.num_pts * 6, out->logical_shape()[-1]);
    }
    TT_FATAL(a.row_width >= 2 * a.num_pts, "row_width must hold 2*num_pts coordinates");
    TT_FATAL(a.num_levels == 4, "wired for 4 FPN levels (writer takes 4 output addresses)");
}

TensorSpec GridPrecomputeOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.out0.tensor_spec();
}

Tensor GridPrecomputeOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.out0;
}

void grid_precompute(
    const Tensor& cgrid, const Tensor& consts,
    Tensor& out0, Tensor& out1, Tensor& out2, Tensor& out3,
    uint32_t num_pts) {
    const auto& gs = cgrid.logical_shape();
    uint32_t num_rows = 1;
    for (int i = 0; i < gs.rank() - 1; i++) {
        num_rows *= gs[i];
    }
    using Op = GridPrecomputeOperation;
    ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_pts = num_pts,
            .num_levels = 4,
            .num_rows = num_rows,
            .row_width = static_cast<uint32_t>(gs[-1]),
            .output_mem_config = out0.memory_config(),
        },
        Op::tensor_args_t{
            .cgrid = cgrid, .consts = consts,
            .out0 = out0, .out1 = out1, .out2 = out2, .out3 = out3});
}

}  // namespace ttnn::prim
