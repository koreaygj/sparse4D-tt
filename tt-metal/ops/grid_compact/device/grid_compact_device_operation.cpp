// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include "grid_compact_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

GridCompactOperation::program_factory_t GridCompactOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GridCompactProgramFactory{};
}

void GridCompactOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    // Q14 fixed point in a UINT16 container — see kps_project_fused for why the
    // coordinate is fixed point and not bf16. ttnn has no INT16; the kernel reinterprets.
    TT_FATAL(t.grid.dtype() == DataType::UINT16, "grid must be UINT16 (Q14 fixed point)");
    TT_FATAL(t.cgrid.dtype() == t.grid.dtype(), "cgrid must match the grid dtype");
    TT_FATAL(t.index.dtype() == DataType::UINT32, "index must be UINT32");
    TT_FATAL(t.flags.dtype() == DataType::BFLOAT16, "flags must be BFLOAT16");
    TT_FATAL(t.grid.layout() == Layout::ROW_MAJOR, "grid must be ROW_MAJOR");
    TT_FATAL(t.cgrid.layout() == Layout::ROW_MAJOR, "cgrid must be ROW_MAJOR");
    TT_FATAL(t.grid.storage_type() == StorageType::DEVICE, "grid must be on device");
    TT_FATAL(t.cgrid.storage_type() == StorageType::DEVICE, "cgrid must be on device");
    TT_FATAL(t.index.storage_type() == StorageType::DEVICE, "index must be on device");
    TT_FATAL(t.flags.storage_type() == StorageType::DEVICE, "flags must be on device");
    TT_FATAL(t.flags.layout() == Layout::ROW_MAJOR, "flags must be ROW_MAJOR");
    TT_FATAL(
        t.flags.logical_shape()[-1] >= a.anchors,
        "flags last dim must be >= anchors ({} < {})",
        t.flags.logical_shape()[-1],
        a.anchors);
    TT_FATAL(a.row_width >= 2 * a.num_pts, "row_width must hold 2*num_pts coordinates");
    if (t.bidx.has_value()) {
        TT_FATAL(t.bidx->dtype() == DataType::UINT32, "bidx must be UINT32");
        TT_FATAL(t.bidx->layout() == Layout::ROW_MAJOR, "bidx must be ROW_MAJOR");
        TT_FATAL(t.bidx->storage_type() == StorageType::DEVICE, "bidx must be on device");
    }
}

TensorSpec GridCompactOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.cgrid.tensor_spec();
}

Tensor GridCompactOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.cgrid;
}

void grid_compact(
    const Tensor& grid, Tensor& cgrid, Tensor& index, Tensor& flags,
    uint32_t num_pts, uint32_t capacity, uint32_t anchors,
    float threshold_x, float threshold_y,
    const std::optional<Tensor>& bidx) {
    const auto& gs = grid.logical_shape();
    const uint32_t row_width = gs[-1];
    uint32_t num_rows = 1;
    for (int i = 0; i < gs.rank() - 1; i++) {
        num_rows *= gs[i];
    }
    // The kernel compares Q14 fixed point, so the threshold becomes an integer. Round it
    // UP: the test is a strict `<`, and rounding down would move the boundary inwards and
    // drop rows whose only in-bounds point sits between two representable coordinates.
    constexpr int32_t GRID_FIXED_SHIFT = 14;  // must match kps_project_fused and grid_sample
    auto to_fixed_ceil = [](float v) -> uint32_t {
        const float s = v * (float)(1 << GRID_FIXED_SHIFT);
        int32_t q = (int32_t)s;
        if ((float)q < s) {
            q += 1;
        }
        if (q > 32767) {
            q = 32767;
        }
        return (uint32_t)q;
    };
    const uint32_t thr_x_bits = to_fixed_ceil(threshold_x);
    const uint32_t thr_y_bits = to_fixed_ceil(threshold_y);

    using Op = GridCompactOperation;
    ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            .num_rows = num_rows,
            .num_pts = num_pts,
            .row_width = row_width,
            .capacity = capacity,
            .anchors = anchors,
            .thr_x_bits = thr_x_bits,
            .thr_y_bits = thr_y_bits,
            .flag_width = static_cast<uint32_t>(flags.logical_shape()[-1]),
            .bidx_width = bidx.has_value() ? static_cast<uint32_t>(bidx->logical_shape()[-1]) : 0u,
            .output_mem_config = cgrid.memory_config(),
        },
        Op::tensor_args_t{
            .grid = grid, .cgrid = cgrid, .index = index, .flags = flags, .bidx = bidx});
}

}  // namespace ttnn::prim
