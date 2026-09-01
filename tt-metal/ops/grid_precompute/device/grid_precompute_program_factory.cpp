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
#include "grid_precompute_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

// One coords tile per core: a core owns at most 32 compacted rows, which is what
// keeps every stage of this op a single-tile pipeline. cap ~1176 rows lands on
// ~37 cores; the arithmetic per core is ~120 tile ops and the writer's scalar
// conversion ~6K elements, so the op prices in the tens of microseconds where
// the reader math it replaces measured ~6.5 ms/frame.
GridPrecomputeProgramFactory::cached_program_t GridPrecomputeProgramFactory::create(
    const GridPrecomputeParams& attrs,
    const GridPrecomputeInputs& t,
    Tensor& /*output_tensor*/) {

    Program program{};
    const uint32_t align = hal::get_dram_alignment();

    constexpr uint32_t ROWS_PER_CORE = 32;
    const uint32_t num_cores = (attrs.num_rows + ROWS_PER_CORE - 1) / ROWS_PER_CORE;
    const auto grid_size = t.cgrid.device()->compute_with_storage_grid_size();
    TT_FATAL(num_cores <= grid_size.x * grid_size.y,
             "grid_precompute: {} rows need {} cores, device has {}",
             attrs.num_rows, num_cores, grid_size.x * grid_size.y);
    const CoreRangeSet core_range = num_cores_to_corerangeset(num_cores, grid_size, true);

    const uint32_t cgrid_page = tt::round_up(attrs.row_width * 2, align);
    const uint32_t out_page = tt::round_up(attrs.num_pts * 6 * 2, align);
    const uint32_t out_tiles = (attrs.num_pts * 6 + 31) / 32;
    constexpr uint32_t f32_tile = 32 * 32 * 4;

    const auto f32_df = tt::DataFormat::Float32;
    const auto u16_df = tt::DataFormat::UInt16;

    auto mk_cb = [&](uint32_t idx, uint32_t total, uint32_t page, tt::DataFormat df) {
        auto cfg = CircularBufferConfig(total, {{idx, df}}).set_page_size(idx, page);
        CreateCircularBuffer(program, core_range, cfg);
    };
    mk_cb(tt::CBIndex::c_0, f32_tile, f32_tile, f32_df);            // coords
    // Constant pack, resident: pushed once, indexed by tile, never popped.
    const uint32_t n_const = 3 * attrs.num_levels + 5 * out_tiles;
    mk_cb(tt::CBIndex::c_1, n_const * f32_tile, f32_tile, f32_df);
    mk_cb(tt::CBIndex::c_3, f32_tile, f32_tile, f32_df);            // F
    mk_cb(tt::CBIndex::c_5, f32_tile, f32_tile, f32_df);            // R
    mk_cb(tt::CBIndex::c_6, f32_tile, f32_tile, f32_df);            // FA0
    mk_cb(tt::CBIndex::c_7, f32_tile, f32_tile, f32_df);            // FA1
    // Output rings hold two levels' worth so the writer's copy of level l
    // overlaps the compute of level l+1. cb_16 is BF16 — the packer does the
    // f32->bf16 conversion in hardware — and cb_17 carries the SFPU-typecast
    // int32 indices.
    constexpr uint32_t bf16_tile = 32 * 32 * 2;
    mk_cb(tt::CBIndex::c_16, 2 * out_tiles * bf16_tile, bf16_tile, tt::DataFormat::Float16_b);
    mk_cb(tt::CBIndex::c_17, 2 * out_tiles * f32_tile, f32_tile, tt::DataFormat::Int32);
    mk_cb(tt::CBIndex::c_24, ROWS_PER_CORE * cgrid_page, ROWS_PER_CORE * cgrid_page, u16_df);  // reader scratch
    mk_cb(tt::CBIndex::c_25, ROWS_PER_CORE * out_page, ROWS_PER_CORE * out_page, u16_df);      // writer stage

    // Reader
    std::vector<uint32_t> reader_ct = {
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_24,
        attrs.num_pts, cgrid_page, attrs.num_levels, out_tiles,
    };
    TensorAccessorArgs(*t.cgrid.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*t.consts.buffer()).append_to(reader_ct);
    auto reader_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/grid_precompute/device/kernels/dataflow/reader_grid_precompute.cpp",
        core_range, ReaderDataMovementConfig(reader_ct));

    // Compute — fp32 dest, and the coords CB unpacks STRAIGHT to dest. The
    // default copy_tile path goes through the srcA register, which holds fp32
    // at ~10 mantissa bits; a raw Q14 coordinate needs 15, and the truncation
    // measured out as floor() flipping a pixel index on ~1% of points (every
    // mismatch sat within 0.005 below an integer). The constants do NOT need
    // the exact path: SCALE is 11*2^-k, BIAS is W/2-0.5 and C is a small
    // integer — all exactly representable in 10 bits — and the selector
    // matrices must stay on the regular unpack path anyway to feed matmul.
    std::vector<UnpackToDestMode> unpack_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_modes[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    auto compute_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/grid_precompute/device/kernels/compute/compute_grid_precompute.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_modes,
            .math_approx_mode = false,
            .compile_args = {},
        });

    // Writer
    std::vector<uint32_t> writer_ct = {
        tt::CBIndex::c_16, tt::CBIndex::c_25,
        attrs.num_pts, attrs.num_levels, out_tiles, out_page,
        tt::CBIndex::c_17,
    };
    // All four outputs are allocated with identical shape/dtype/memory config,
    // so one accessor arg set serves them all; only the base address differs.
    TensorAccessorArgs(*t.out0.buffer()).append_to(writer_ct);
    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/grid_precompute/device/kernels/dataflow/writer_grid_precompute.cpp",
        core_range, WriterDataMovementConfig(writer_ct));

    auto cores = corerange_to_cores(core_range, num_cores, true);
    for (uint32_t i = 0; i < num_cores; i++) {
        const uint32_t row_start = i * ROWS_PER_CORE;
        const uint32_t rows = std::min(ROWS_PER_CORE, attrs.num_rows - row_start);
        SetRuntimeArgs(program, reader_id, cores[i],
                       {t.cgrid.buffer()->address(), t.consts.buffer()->address(), row_start, rows});
        SetRuntimeArgs(program, compute_id, cores[i], {attrs.num_levels, out_tiles});
        SetRuntimeArgs(program, writer_id, cores[i],
                       {t.out0.buffer()->address(), t.out1.buffer()->address(),
                        t.out2.buffer()->address(), t.out3.buffer()->address(),
                        row_start, rows});
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{.reader_id = reader_id, .writer_id = writer_id, .cores = std::move(cores)}};
}

void GridPrecomputeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const GridPrecomputeParams&,
    const GridPrecomputeInputs& t,
    Tensor& /*output_tensor*/) {
    auto& prog = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    for (const auto& core : sv.cores) {
        auto& r = GetRuntimeArgs(prog, sv.reader_id, core);
        r[0] = t.cgrid.buffer()->address();
        r[1] = t.consts.buffer()->address();
        auto& w = GetRuntimeArgs(prog, sv.writer_id, core);
        w[0] = t.out0.buffer()->address();
        w[1] = t.out1.buffer()->address();
        w[2] = t.out2.buffer()->address();
        w[3] = t.out3.buffer()->address();
    }
}

}  // namespace ttnn::prim
