// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// 3-kernel TILE-based program factory for kps_project_fused
// Reader: construct rotation/projection tiles from ROW_MAJOR inputs
// Compute: matmul_tiles for rotation and projection (FPU)
// Writer: scalar perspective divide + grid normalize

#include <cstdint>
#include "tt-metalium/work_split.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "kps_project_fused_tile_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

KpsProjectFusedTileProgramFactory::cached_program_t KpsProjectFusedTileProgramFactory::create(
    const KpsProjectFusedParams& attrs,
    const KpsProjectFusedInputs& t,
    Tensor& output_tensor) {

    Program program{};
    IDevice* const device = output_tensor.device();

    const uint32_t n = t.key_points.logical_shape()[0];
    const uint32_t num_pts = attrs.num_pts;
    const uint32_t nc = attrs.num_cams;

    // Page sizes for ROW_MAJOR inputs (same as scalar version)
    const uint32_t align = t.key_points.buffer()->buffer_type() == BufferType::DRAM
                           ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const uint32_t kp_elem = t.key_points.element_size();
    const uint32_t anc_elem = t.anchor.element_size();
    const uint32_t kp_page_size = tt::round_up(3 * kp_elem, align);
    const uint32_t anchor_page_size = tt::round_up(11 * anc_elem, align);
    const uint32_t proj_page_size = tt::round_up(4 * sizeof(float), align);
    const uint32_t wh_page_size = tt::round_up(2 * sizeof(float), align);
    const uint32_t out_page_size = tt::round_up(num_pts * 2 * sizeof(float), align);

    // TILE sizes
    const uint32_t bf16_tile_size = 2048;  // 32×32 × 2 bytes
    const uint32_t f32_tile_size = 4096;   // 32×32 × 4 bytes

    // Work distribution
    const auto compute_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, n_anchors_1, n_anchors_2] =
        split_work_to_cores(compute_grid_size, n);
    std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, num_cores, true);

    // Circular Buffers
    uint32_t cb_idx = tt::CBIndex::c_0;
    const auto bf16_df = tt::DataFormat::Float16_b;
    const auto f32_df = tt::DataFormat::Float32;

    // CB0: key_points TILE [32,32] bf16 (1 tile)
    const auto [cb_kp, _kp] = create_cb(cb_idx++, program, all_cores, bf16_tile_size, 1, bf16_df);
    // CB1: rotation matrix TILE [32,32] bf16
    const auto [cb_rot, _rot] = create_cb(cb_idx++, program, all_cores, bf16_tile_size, 1, bf16_df);
    // CB2: center broadcast TILE [32,32] bf16
    const auto [cb_center, _ctr] = create_cb(cb_idx++, program, all_cores, bf16_tile_size, 1, bf16_df);
    // CB3: ones_col3 TILE [32,32] f32
    const auto [cb_ones, _ones] = create_cb(cb_idx++, program, all_cores, f32_tile_size, 1, f32_df);
    // CB4: projection P^T TILE [32,32] f32 (per camera, streamed)
    const auto [cb_proj, _proj] = create_cb(cb_idx++, program, all_cores, f32_tile_size, 1, f32_df);
    // CB5: intermediate rotated TILE bf16 (used by compute kernel)
    create_cb(cb_idx++, program, all_cores, bf16_tile_size, 1, bf16_df);
    // CB6: intermediate translated TILE bf16 (used by compute kernel)
    create_cb(cb_idx++, program, all_cores, bf16_tile_size, 1, bf16_df);
    // CB7: intermediate homogeneous TILE f32 (used by compute kernel)
    create_cb(cb_idx++, program, all_cores, f32_tile_size, 1, f32_df);

    // CB16: projected output TILE f32
    const auto [cb_out, _out] = create_cb(tt::CBIndex::c_16, program, all_cores, f32_tile_size, 1, f32_df);

    // CB17: writer scratch for image_wh + output page
    const uint32_t wh_scratch_size = tt::round_up(nc * wh_page_size + out_page_size, align);
    const auto [cb_wh_scratch, _wh] = create_cb(tt::CBIndex::c_17, program, all_cores, wh_scratch_size, 1, f32_df);

    // CB18: reader scratch for raw DRAM pages (32-byte zero region + anchor + kp + proj)
    const uint32_t reader_scratch_size = tt::round_up(
        32 + anchor_page_size + num_pts * kp_page_size + 4 * proj_page_size, align);
    create_cb(tt::CBIndex::c_18, program, all_cores, reader_scratch_size, 1, bf16_df);

    // Reader kernel
    std::vector<uint32_t> reader_ct_args = {
        cb_kp, cb_rot, cb_center, cb_ones, cb_proj,
        nc, num_pts,
        kp_page_size, anchor_page_size, proj_page_size, wh_page_size,
        bf16_tile_size, f32_tile_size,
    };
    TensorAccessorArgs(*t.key_points.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*t.anchor.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*t.projection_mat.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*t.image_wh.buffer()).append_to(reader_ct_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/kps_project_fused/device/kernels/dataflow/reader_kps_project_tile.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    // Compute kernel
    std::vector<uint32_t> compute_ct_args = {};  // runtime args only
    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/kps_project_fused/device/kernels/compute/compute_kps_project.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .compile_args = compute_ct_args,
        });

    // Writer kernel
    std::vector<uint32_t> writer_ct_args = {
        cb_out, out_page_size, nc, num_pts, wh_page_size,
        cb_wh_scratch,
    };
    TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*t.image_wh.buffer()).append_to(writer_ct_args);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/kps_project_fused/device/kernels/dataflow/writer_kps_project_tile.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // Runtime args per core
    uint32_t anchor_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        const uint32_t core_anchors = core_group_1.contains(core) ? n_anchors_1 : n_anchors_2;

        SetRuntimeArgs(program, reader_kernel_id, core, {
            t.key_points.buffer()->address(),
            t.anchor.buffer()->address(),
            t.projection_mat.buffer()->address(),
            t.image_wh.buffer()->address(),
            core_anchors,
            anchor_offset,
        });

        SetRuntimeArgs(program, compute_kernel_id, core, {
            core_anchors,
            nc,
        });

        SetRuntimeArgs(program, writer_kernel_id, core, {
            output_tensor.buffer()->address(),
            t.image_wh.buffer()->address(),
            core_anchors,
            anchor_offset,
            n,  // total anchors
        });

        anchor_offset += core_anchors;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .num_cores = num_cores,
            .logical_cores = logical_cores,
        }};
}

void KpsProjectFusedTileProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const KpsProjectFusedParams&,
    const KpsProjectFusedInputs& t,
    Tensor& output_tensor) {
    auto& prog = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    for (uint32_t i = 0; i < sv.num_cores; i++) {
        auto& r = GetRuntimeArgs(prog, sv.reader_kernel_id, sv.logical_cores[i]);
        r[0] = t.key_points.buffer()->address();
        r[1] = t.anchor.buffer()->address();
        r[2] = t.projection_mat.buffer()->address();
        r[3] = t.image_wh.buffer()->address();
        auto& w = GetRuntimeArgs(prog, sv.writer_kernel_id, sv.logical_cores[i]);
        w[0] = output_tensor.buffer()->address();
        w[1] = t.image_wh.buffer()->address();
    }
}

}  // namespace ttnn::prim
