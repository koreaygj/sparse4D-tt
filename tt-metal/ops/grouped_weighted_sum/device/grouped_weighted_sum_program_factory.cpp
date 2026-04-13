// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/work_split.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/hal.hpp>
#include "grouped_weighted_sum_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

GroupedWeightedSumProgramFactory::cached_program_t GroupedWeightedSumProgramFactory::create(
    const GroupedWeightedSumParams& attrs,
    const GroupedWeightedSumInputs& t,
    Tensor& output_tensor) {

    Program program{};
    IDevice* const device = output_tensor.device();

    const uint32_t n = t.features.logical_shape()[0];    // CLP (batch/reduction)
    const uint32_t clp = t.features.logical_shape()[1];  // N (output anchors)
    const uint32_t G = attrs.num_groups;
    const uint32_t D = attrs.group_dims;
    const uint32_t embed_dims = G * D;

    const uint32_t clp_padded = tt::round_up(clp, 32);
    const uint32_t num_output_tile_rows = clp_padded / 32;  // N_TR
    const uint32_t num_feat_tile_cols = embed_dims / 32;     // 8

    // 2-phase reduction for more parallelism
    const uint32_t num_chunks = 2;
    const uint32_t chunk_size = (n + num_chunks - 1) / num_chunks;
    const uint32_t num_work_units = num_output_tile_rows * num_chunks;

    const auto feat_df = datatype_to_dataformat_converter(t.features.dtype());
    const auto wt_df = datatype_to_dataformat_converter(t.weights.dtype());
    const auto out_df = datatype_to_dataformat_converter(output_tensor.dtype());

    const uint32_t feat_tile_bytes = tile_size(feat_df);
    const uint32_t wt_tile_bytes = tile_size(wt_df);
    const uint32_t out_tile_bytes = tile_size(out_df);

    const auto compute_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, nw1, nw2] =
        split_work_to_cores(compute_grid_size, num_work_units);
    std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, num_cores, true);

    uint32_t cb_idx = tt::CBIndex::c_0;
    const auto [feat_cb, _f] = create_cb(cb_idx++, program, all_cores, feat_tile_bytes, G * 2, feat_df);
    const auto [wt_raw_cb, _wr] = create_cb(cb_idx++, program, all_cores, wt_tile_bytes, 1, wt_df);
    const auto [wt_col_cb, _wc] = create_cb(cb_idx++, program, all_cores, wt_tile_bytes, G * 2, wt_df);
    const auto [out_cb, _o] = create_cb(cb_idx++, program, all_cores, out_tile_bytes, G, out_df);

    const bool rm_mode = (t.features.layout() == Layout::ROW_MAJOR);
    const uint32_t rm_stick_size = embed_dims * t.features.element_size();
    const uint32_t N_total = rm_mode ? t.features.logical_shape()[1] : 0;

    // Reader
    std::vector<uint32_t> reader_ct_args = {
        feat_cb, wt_raw_cb, wt_col_cb,
        G,                                               // 3
        num_feat_tile_cols,                              // 4: FEAT_TC
        rm_mode ? rm_stick_size : feat_tile_bytes,       // 5: page_size (stick for RM, tile for TILE)
        wt_tile_bytes,                                   // 6
        num_output_tile_rows,                            // 7: N_TR
    };
    TensorAccessorArgs(*t.features.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*t.weights.buffer()).append_to(reader_ct_args);
    // RM mode args (after tensor accessors)
    reader_ct_args.push_back(rm_mode ? 1U : 0U);  // RM_MODE
    reader_ct_args.push_back(N_total);              // RM_N
    reader_ct_args.push_back(rm_mode ? rm_stick_size : 0U); // RM_STICK_SZ

    auto reader_kernel_id = CreateKernel(program,
        "ttnn/cpp/ttnn/operations/pool/grouped_weighted_sum/device/kernels/dataflow/reader_grouped_weighted_sum.cpp",
        all_cores, ReaderDataMovementConfig(reader_ct_args));

    // Compute — TILE mode: direct bcast. RM mode: tilize first then bcast.
    uint32_t tile_cb = 0;
    if (rm_mode) {
        auto [_tc, _tcx] = create_cb(cb_idx++, program, all_cores, feat_tile_bytes, G, feat_df);
        tile_cb = _tc;
    }
    std::vector<uint32_t> compute_ct_args = {
        feat_cb, wt_col_cb, out_cb,
        G,                                    // 3
        rm_mode ? 1U : 0U,                   // 4: RM_MODE
        rm_mode ? tile_cb : feat_cb,          // 5: tile_cb (intermediate for RM)
    };
    auto compute_kernel_id = CreateKernel(program,
        "ttnn/cpp/ttnn/operations/pool/grouped_weighted_sum/device/kernels/compute/compute_grouped_weighted_sum.cpp",
        all_cores, ComputeConfig{.math_fidelity=MathFidelity::HiFi4, .fp32_dest_acc_en=false, .compile_args=compute_ct_args});

    // Writer
    std::vector<uint32_t> writer_ct_args = {
        out_cb, out_tile_bytes, G,
        num_output_tile_rows,     // 3: N_TR for output addressing
    };
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct_args);

    auto writer_kernel_id = CreateKernel(program,
        "ttnn/cpp/ttnn/operations/pool/grouped_weighted_sum/device/kernels/dataflow/writer_grouped_weighted_sum.cpp",
        all_cores, WriterDataMovementConfig(writer_ct_args));

    // Runtime args: each work unit = (tile_row, chunk_id)
    uint32_t wu_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        const uint32_t core_wus = core_group_1.contains(core) ? nw1 : nw2;

        SetRuntimeArgs(program, reader_kernel_id, core, {
            t.features.buffer()->address(),
            t.weights.buffer()->address(),
            core_wus,           // num work units
            wu_offset,          // start work unit
            num_output_tile_rows, // for deriving tile_row from WU
            chunk_size,         // CLP batches per chunk
            n,                  // total CLP batches
        });

        SetRuntimeArgs(program, compute_kernel_id, core, {
            core_wus,           // num work units
            chunk_size,         // max CLP per chunk (for compute loop)
            n,                  // total CLP
        });

        SetRuntimeArgs(program, writer_kernel_id, core, {
            output_tensor.buffer()->address(),
            core_wus,
            wu_offset,
            num_output_tile_rows,
            num_chunks,
        });

        wu_offset += core_wus;
    }

    return cached_program_t{std::move(program),
        shared_variables_t{.reader_kernel_id=reader_kernel_id, .writer_kernel_id=writer_kernel_id,
                           .num_cores=num_cores, .logical_cores=logical_cores}};
}

void GroupedWeightedSumProgramFactory::override_runtime_arguments(
    cached_program_t& cp, const GroupedWeightedSumParams&,
    const GroupedWeightedSumInputs& t, Tensor& out) {
    auto& p = cp.program; const auto& sv = cp.shared_variables;
    for (uint32_t i = 0; i < sv.num_cores; i++) {
        auto& r = GetRuntimeArgs(p, sv.reader_kernel_id, sv.logical_cores[i]);
        r[0] = t.features.buffer()->address(); r[1] = t.weights.buffer()->address();
        GetRuntimeArgs(p, sv.writer_kernel_id, sv.logical_cores[i])[0] = out.buffer()->address();
    }
}
}
