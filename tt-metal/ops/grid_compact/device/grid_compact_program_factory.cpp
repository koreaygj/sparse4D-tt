// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "grid_compact_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

// Single core on purpose: the pass is a sequential scan with a running output
// counter, which across cores would need a prefix sum. Measured alternative --
// giving each core its own fixed slot budget so no core has to talk -- has to size
// for the busiest core and yields only 1.5x compaction versus 2.4x pooled.
GridCompactProgramFactory::cached_program_t GridCompactProgramFactory::create(
    const GridCompactParams& attrs,
    const GridCompactInputs& t,
    Tensor& /*output_tensor*/) {

    Program program{};
    const CoreCoord core{0, 0};
    const CoreRange core_range(core, core);

    constexpr uint32_t BATCH = 64;   // rows fetched per barrier
    const uint32_t row_bytes = attrs.row_width * sizeof(float);
    // L1 strides, 32 B aligned: NOC endpoints must be, and a 26-float row is not.
    const uint32_t row_stride = ((row_bytes + 31) / 32) * 32;
    const uint32_t num_cams = attrs.num_rows / attrs.anchors;

    const bool pooled = t.bidx.has_value();
    const uint32_t idx_bytes = (pooled ? attrs.capacity : num_cams * attrs.capacity) * sizeof(uint32_t);
    const uint32_t bidx_stride = pooled ? attrs.bidx_width * sizeof(uint32_t) : 0;
    const uint32_t bidx_bytes = pooled ? attrs.capacity * bidx_stride : 0;

    constexpr uint32_t ROW_CB = tt::CBIndex::c_0;
    constexpr uint32_t IDX_CB = tt::CBIndex::c_1;
    constexpr uint32_t FLG_CB = tt::CBIndex::c_2;
    const uint32_t flag_stride = ((attrs.flag_width * 2 + 31) / 32) * 32;
    const uint32_t flag_bytes = num_cams * flag_stride;

    auto row_cb_cfg = CircularBufferConfig(row_stride * BATCH, {{ROW_CB, DataFormat::Float32}})
                          .set_page_size(ROW_CB, row_stride);
    CreateCircularBuffer(program, core_range, row_cb_cfg);
    auto idx_cb_cfg = CircularBufferConfig(idx_bytes, {{IDX_CB, DataFormat::UInt32}})
                          .set_page_size(IDX_CB, idx_bytes);
    CreateCircularBuffer(program, core_range, idx_cb_cfg);
    auto flg_cb_cfg = CircularBufferConfig(flag_bytes, {{FLG_CB, DataFormat::Float16_b}})
                          .set_page_size(FLG_CB, flag_bytes);
    CreateCircularBuffer(program, core_range, flg_cb_cfg);
    constexpr uint32_t BIDX_CB = tt::CBIndex::c_3;
    if (pooled) {
        auto bidx_cb_cfg = CircularBufferConfig(bidx_bytes, {{BIDX_CB, DataFormat::UInt32}})
                               .set_page_size(BIDX_CB, bidx_bytes);
        CreateCircularBuffer(program, core_range, bidx_cb_cfg);
    }

    std::vector<uint32_t> ct_args = {
        attrs.num_rows,    // 0
        attrs.num_pts,     // 1
        attrs.row_width,   // 2
        attrs.capacity,    // 3
        attrs.anchors,     // 4
        attrs.thr_x_bits,  // 5
        ROW_CB,            // 6
        IDX_CB,            // 7
        BATCH,             // 8
        FLG_CB,            // 9
        attrs.flag_width,  // 10
        attrs.thr_y_bits,  // 11
        pooled ? 1u : 0u,  // 12
        BIDX_CB,           // 13
        attrs.bidx_width,  // 14
    };
    TensorAccessorArgs(*t.grid.buffer()).append_to(ct_args);
    TensorAccessorArgs(*t.cgrid.buffer()).append_to(ct_args);
    TensorAccessorArgs(*t.index.buffer()).append_to(ct_args);
    TensorAccessorArgs(*t.flags.buffer()).append_to(ct_args);
    if (pooled) {
        TensorAccessorArgs(*t.bidx->buffer()).append_to(ct_args);
    }

    KernelHandle kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/grid_compact/device/kernels/dataflow/grid_compact.cpp",
        core_range,
        WriterDataMovementConfig(ct_args));

    SetRuntimeArgs(program, kernel_id, core, {
        t.grid.buffer()->address(),
        t.cgrid.buffer()->address(),
        t.index.buffer()->address(),
        t.flags.buffer()->address(),
        pooled ? t.bidx->buffer()->address() : 0u});

    return cached_program_t{
        std::move(program),
        shared_variables_t{.kernel_id = kernel_id, .core = core}};
}

void GridCompactProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const GridCompactParams&,
    const GridCompactInputs& t,
    Tensor& /*output_tensor*/) {
    auto& prog = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    auto& r = GetRuntimeArgs(prog, sv.kernel_id, sv.core);
    r[0] = t.grid.buffer()->address();
    r[1] = t.cgrid.buffer()->address();
    r[2] = t.index.buffer()->address();
    r[3] = t.flags.buffer()->address();
    if (t.bidx.has_value()) {
        r[4] = t.bidx->buffer()->address();
    }
}

}  // namespace ttnn::prim
