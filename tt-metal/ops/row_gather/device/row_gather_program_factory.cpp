// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "row_gather_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

RowGatherProgramFactory::cached_program_t RowGatherProgramFactory::create(
    const RowGatherParams& attrs,
    const RowGatherInputs& t,
    Tensor& /*output_tensor*/) {

    Program program{};

    const bool has_b = t.src_b.has_value();
    const bool a_rm = t.src_a.layout() == Layout::ROW_MAJOR;
    const uint32_t a_stick = t.src_a.logical_shape()[-1] * t.src_a.element_size();
    const uint32_t ct_a = t.src_a.padded_shape()[-1] / constants::TILE_WIDTH;
    const uint32_t ct_b = has_b ? t.src_b->padded_shape()[-1] / constants::TILE_WIDTH : 0;
    const uint32_t ea = t.src_a.element_size();
    const uint32_t eb = has_b ? t.src_b->element_size() : 4;

    const auto grid = t.src_a.device()->compute_with_storage_grid_size();
    uint32_t num_cores = std::min<uint32_t>(grid.x * grid.y, (attrs.k + 7) / 8);
    const uint32_t rows_base = attrs.k / num_cores;
    const uint32_t rows_rem = attrs.k % num_cores;
    const CoreRangeSet core_range = num_cores_to_corerangeset(num_cores, grid, true);

    constexpr uint32_t CB_IDX = tt::CBIndex::c_0;
    constexpr uint32_t CB_ROW = tt::CBIndex::c_1;
    const uint32_t idx_bytes = ((attrs.k * 4 + 31) / 32) * 32;
    const uint32_t row_bytes =
        (a_rm ? ((a_stick + 31) & ~31u) : 32 * ct_a * ea) + 32 * ct_b * eb;

    auto idx_cfg = CircularBufferConfig(idx_bytes, {{CB_IDX, DataFormat::UInt32}})
                       .set_page_size(CB_IDX, idx_bytes);
    CreateCircularBuffer(program, core_range, idx_cfg);
    auto row_cfg = CircularBufferConfig(row_bytes, {{CB_ROW, DataFormat::UInt8}})
                       .set_page_size(CB_ROW, row_bytes);
    CreateCircularBuffer(program, core_range, row_cfg);

    std::vector<uint32_t> ct_args = {attrs.k, ct_a, ct_b, ea, eb, CB_IDX, CB_ROW,
                                     a_rm ? 1u : 0u, a_stick};
    TensorAccessorArgs(*t.src_a.buffer()).append_to(ct_args);
    TensorAccessorArgs(has_b ? *t.src_b->buffer() : *t.src_a.buffer()).append_to(ct_args);
    TensorAccessorArgs(*t.indices.buffer()).append_to(ct_args);
    TensorAccessorArgs(*t.out_a.buffer()).append_to(ct_args);
    TensorAccessorArgs(has_b ? *t.out_b->buffer() : *t.out_a.buffer()).append_to(ct_args);

    KernelHandle kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/row_gather/device/kernels/dataflow/row_gather.cpp",
        core_range, WriterDataMovementConfig(ct_args));

    auto cores = corerange_to_cores(core_range, num_cores, true);
    uint32_t r0 = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const uint32_t my_rows = rows_base + (i < rows_rem ? 1 : 0);
        SetRuntimeArgs(program, kernel_id, cores[i], {
            t.src_a.buffer()->address(),
            has_b ? t.src_b->buffer()->address() : 0u,
            t.indices.buffer()->address(),
            t.out_a.buffer()->address(),
            has_b ? t.out_b->buffer()->address() : 0u,
            r0, my_rows});
        r0 += my_rows;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{.kernel_id = kernel_id, .cores = std::move(cores)}};
}

void RowGatherProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RowGatherParams&,
    const RowGatherInputs& t,
    Tensor& /*output_tensor*/) {
    auto& prog = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    for (const auto& c : sv.cores) {
        auto& ra = GetRuntimeArgs(prog, sv.kernel_id, c);
        ra[0] = t.src_a.buffer()->address();
        ra[1] = t.src_b.has_value() ? t.src_b->buffer()->address() : 0u;
        ra[2] = t.indices.buffer()->address();
        ra[3] = t.out_a.buffer()->address();
        ra[4] = t.out_b.has_value() ? t.out_b->buffer()->address() : 0u;
    }
}

}  // namespace ttnn::prim
