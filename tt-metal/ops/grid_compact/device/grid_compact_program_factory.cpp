// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include "tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include "grid_compact_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

// The bounds test runs on every core, the writes on one. The test is ~58% of the runtime
// at the production shape and has no ordering dependency; the writes need a running slot
// counter, which across cores would need a prefix sum. Giving each core its own fixed slot
// budget instead -- so no core has to talk -- was measured and rejected: it has to size for
// the busiest core, which is most of what pooling across cameras exists to avoid.
GridCompactProgramFactory::cached_program_t GridCompactProgramFactory::create(
    const GridCompactParams& attrs,
    const GridCompactInputs& t,
    Tensor& /*output_tensor*/) {

    Program program{};
    const CoreCoord core{0, 0};

    constexpr uint32_t BATCH = 64;   // rows fetched per barrier
    const uint32_t row_bytes = attrs.row_width * t.grid.element_size();
    // L1 strides, 32 B aligned: NOC endpoints must be, and a 26-float row is not.
    const uint32_t row_stride = ((row_bytes + 31) / 32) * 32;
    const uint32_t num_cams = attrs.num_rows / attrs.anchors;

    // Phase-1 fan-out. Fewer rows per core than 8 is not worth the relay, and the mask
    // block is padded to 32 B so each core's relay lands on a NOC-legal boundary.
    const auto grid_size = t.grid.device()->compute_with_storage_grid_size();
    const uint32_t max_cores = grid_size.x * grid_size.y;
    uint32_t num_cores = std::min<uint32_t>(max_cores, (attrs.num_rows + 7) / 8);
    if (num_cores == 0) {
        num_cores = 1;
    }
    const uint32_t rows_per_core = (attrs.num_rows + num_cores - 1) / num_cores;
    // Recompute: ceil division can leave the last cores with no rows at all.
    num_cores = (attrs.num_rows + rows_per_core - 1) / rows_per_core;
    const uint32_t mask_blk = ((rows_per_core + 31) / 32) * 32;
    const CoreRangeSet core_range = num_cores_to_corerangeset(num_cores, grid_size, true);
    const auto c0_physical = t.grid.device()->worker_core_from_logical_core(core);
    // ROW_CB has to hold whichever pass fetches more rows at once.
    const uint32_t row_cb_rows = std::max<uint32_t>(BATCH, rows_per_core);

    const bool pooled = t.bidx.has_value();
    const uint32_t idx_bytes = (pooled ? attrs.capacity : num_cams * attrs.capacity) * sizeof(uint32_t);
    const uint32_t bidx_stride = pooled ? attrs.bidx_width * sizeof(uint32_t) : 0;

    // Indices are ordered so the two buffers every core needs come first. CB addresses are
    // packed per core by index, so ROW_CB and MASK_CB land at the same L1 offset on every
    // core — which is what lets a worker relay its mask block to core 0 by address.
    constexpr uint32_t ROW_CB  = tt::CBIndex::c_0;
    constexpr uint32_t MASK_CB = tt::CBIndex::c_1;
    constexpr uint32_t IDX_CB  = tt::CBIndex::c_2;
    constexpr uint32_t FLG_CB  = tt::CBIndex::c_3;
    constexpr uint32_t BIDX_CB = tt::CBIndex::c_4;
    const uint32_t flag_stride = ((attrs.flag_width * 2 + 31) / 32) * 32;
    const uint32_t flag_bytes = num_cams * flag_stride;
    const uint32_t mask_bytes = num_cores * mask_blk;

    auto row_cb_cfg = CircularBufferConfig(row_stride * row_cb_rows, {{ROW_CB, DataFormat::UInt16}})
                          .set_page_size(ROW_CB, row_stride);
    CreateCircularBuffer(program, core_range, row_cb_cfg);
    auto mask_cb_cfg = CircularBufferConfig(mask_bytes, {{MASK_CB, DataFormat::UInt8}})
                           .set_page_size(MASK_CB, mask_bytes);
    CreateCircularBuffer(program, core_range, mask_cb_cfg);
    auto idx_cb_cfg = CircularBufferConfig(idx_bytes, {{IDX_CB, DataFormat::UInt32}})
                          .set_page_size(IDX_CB, idx_bytes);
    CreateCircularBuffer(program, core_range, idx_cb_cfg);
    auto flg_cb_cfg = CircularBufferConfig(flag_bytes, {{FLG_CB, DataFormat::Float16_b}})
                          .set_page_size(FLG_CB, flag_bytes);
    CreateCircularBuffer(program, core_range, flg_cb_cfg);
    if (pooled) {
        // Only one staged row per camera, not the whole CAP x bidx_width buffer: every row
        // is (camera, 0, 0, ...), so NUM_CAMS distinct rows cover every slot.
        const uint32_t stage_bytes = num_cams * bidx_stride;
        auto bidx_cb_cfg = CircularBufferConfig(stage_bytes, {{BIDX_CB, DataFormat::UInt32}})
                               .set_page_size(BIDX_CB, stage_bytes);
        CreateCircularBuffer(program, core_range, bidx_cb_cfg);
    }

    const uint32_t sem_id = CreateSemaphore(program, core_range, 0);

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
        num_cores,         // 15
        rows_per_core,     // 16
        mask_blk,          // 17
        MASK_CB,           // 18
        sem_id,            // 19
        static_cast<uint32_t>(c0_physical.x),  // 20
        static_cast<uint32_t>(c0_physical.y),  // 21
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

    // The addresses are identical on every core, so they are COMMON args — that keeps the
    // per-dispatch rewrite at 5 words instead of 5 x num_cores. Only the core's own id is
    // per-core, and it never changes.
    SetCommonRuntimeArgs(program, kernel_id, {
        t.grid.buffer()->address(),
        t.cgrid.buffer()->address(),
        t.index.buffer()->address(),
        t.flags.buffer()->address(),
        pooled ? t.bidx->buffer()->address() : 0u});

    // row_wise, so cores[0] is logical (0, 0) — the core the kernel writes from.
    auto cores = corerange_to_cores(core_range, num_cores, true);
    for (uint32_t i = 0; i < num_cores; i++) {
        SetRuntimeArgs(program, kernel_id, cores[i], {i});
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{.kernel_id = kernel_id, .cores = std::move(cores)}};
}

void GridCompactProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const GridCompactParams&,
    const GridCompactInputs& t,
    Tensor& /*output_tensor*/) {
    auto& prog = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    // One shared copy for every core: the per-core args hold only the core id, which the
    // buffer addresses cannot change.
    auto& r = GetCommonRuntimeArgs(prog, sv.kernel_id);
    r[0] = t.grid.buffer()->address();
    r[1] = t.cgrid.buffer()->address();
    r[2] = t.index.buffer()->address();
    r[3] = t.flags.buffer()->address();
    if (t.bidx.has_value()) {
        r[4] = t.bidx->buffer()->address();
    }
}

}  // namespace ttnn::prim
