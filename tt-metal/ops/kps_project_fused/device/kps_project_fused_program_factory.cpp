// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/work_split.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "kps_project_fused_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

KpsProjectFusedProgramFactory::cached_program_t KpsProjectFusedProgramFactory::create(
    const KpsProjectFusedParams& attrs,
    const KpsProjectFusedInputs& t,
    Tensor& output_tensor) {

    Program program{};
    IDevice* const device = output_tensor.device();

    // Standard & fused precompute: key_points is [N, pts, 3], standalone precompute: [nc, N, 1, pts*2]
    const bool standalone_precompute = attrs.precompute_grid && t.key_points.logical_shape().rank() == 4;
    const uint32_t n = standalone_precompute
        ? t.key_points.logical_shape()[1]
        : t.key_points.logical_shape()[0];       // num anchors
    const uint32_t num_pts = attrs.num_pts;                    // 13
    const uint32_t nc = attrs.num_cams;                        // 3 or 6

    // Page sizes: use actual element_size from tensor (bf16=2, f32=4)
    const uint32_t align = t.key_points.buffer()->buffer_type() == BufferType::DRAM
                           ? hal::get_dram_alignment() : hal::get_l1_alignment();
    const uint32_t kp_elem = t.key_points.element_size();      // bf16=2 or f32=4
    const uint32_t anc_elem = t.anchor.element_size();
    const uint32_t kp_page_size = tt::round_up(3 * kp_elem, align);
    const uint32_t anchor_page_size = tt::round_up(11 * anc_elem, align);
    const uint32_t proj_page_size = tt::round_up(4 * sizeof(float), align);     // proj_mat always f32
    const uint32_t wh_page_size = tt::round_up(2 * sizeof(float), align);       // image_wh always f32
    const uint32_t NL = attrs.precompute_grid ? attrs.spatial_shapes.size() : 0;
    const uint32_t out_elem_size = (attrs.precompute_grid && !standalone_precompute) ? sizeof(uint16_t) : sizeof(float);
    const uint32_t out_elems_per_page = attrs.precompute_grid ? num_pts * 6 : num_pts * 2;
    const uint32_t out_page_size = tt::round_up(out_elems_per_page * out_elem_size, align);

    // Work distribution: split anchors across cores
    const auto compute_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, n_anchors_1, n_anchors_2] =
        split_work_to_cores(compute_grid_size, n);
    std::vector<CoreCoord> logical_cores = corerange_to_cores(all_cores, num_cores, true);

    // Circular Buffers
    uint32_t cb_idx = tt::CBIndex::c_0;
    const auto f32_df = tt::DataFormat::Float32;

    const auto kp_df = datatype_to_dataformat_converter(t.key_points.dtype());
    const auto anc_df = datatype_to_dataformat_converter(t.anchor.dtype());

    // CB0: key_points scratch — num_pts pages of kp_page_size
    const uint32_t kp_cb_size = num_pts * kp_page_size;
    const auto [kp_cb_index, kp_cb_handle] =
        create_cb(cb_idx++, program, all_cores, kp_cb_size, 1, kp_df);

    // CB1: anchor scratch — 1 page of anchor_page_size
    const auto [anchor_cb_index, anchor_cb_handle] =
        create_cb(cb_idx++, program, all_cores, anchor_page_size, 1, anc_df);

    // CB2: proj_mat scratch — nc * 4 rows × proj_page_size each (page-aligned)
    const uint32_t proj_cb_size = tt::round_up(nc * 4 * proj_page_size, align);
    const auto [proj_cb_index, proj_cb_handle] =
        create_cb(cb_idx++, program, all_cores, proj_cb_size, 1, f32_df);

    // CB3: image_wh scratch — nc pages × wh_page_size each
    const uint32_t wh_cb_size = tt::round_up(nc * wh_page_size, align);
    const auto [wh_cb_index, wh_cb_handle] =
        create_cb(cb_idx++, program, all_cores, wh_cb_size, 1, f32_df);

    KernelHandle reader_kernel_id, writer_kernel_id;
    const bool fused_precompute = attrs.precompute_grid && t.key_points.logical_shape().rank() == 3;

    // CB4: output scratch
    const uint32_t out_cb_pages = fused_precompute ? nc * NL : ((attrs.precompute_grid && !fused_precompute) ? nc * NL * 2 : nc);
    const auto out_df = (attrs.precompute_grid && !standalone_precompute) ? tt::DataFormat::Float16_b : f32_df;
    const auto [out_cb_index, out_cb_handle] =
        create_cb(cb_idx++, program, all_cores, out_page_size, out_cb_pages, out_df);

    if (attrs.precompute_grid && !fused_precompute) {
        // ===== Standalone precompute mode: lightweight grid conversion =====
        // Input: key_points is actually [nc, N, 1, pts*2] normalized grid f32
        // Output: [nc*NL, N, 1, pts*6] bf16 precomputed grid
        const uint32_t grid_input_page_size = tt::round_up(num_pts * 2 * sizeof(float), align);

        // Grid input CB (reuse kp_cb slot)
        const auto [grid_in_cb, _gi] = create_cb(cb_idx++, program, all_cores, grid_input_page_size, 1, f32_df);

        std::vector<uint32_t> precompute_ct_args = {
            grid_in_cb,         // 0: grid_cb_index
            out_cb_index,       // 1: out_cb_index
            nc,                 // 2: NC
            num_pts,            // 3: NUM_PTS
            grid_input_page_size, // 4: grid_page_size
            out_page_size,      // 5: out_page_size
            NL,                 // 6: NL
        };
        for (auto& [h, w] : attrs.spatial_shapes) {
            precompute_ct_args.push_back(h);
            precompute_ct_args.push_back(w);
        }
        for (uint32_t i = attrs.spatial_shapes.size(); i < 4; i++) {
            precompute_ct_args.push_back(0);
            precompute_ct_args.push_back(0);
        }
        TensorAccessorArgs(*t.key_points.buffer()).append_to(precompute_ct_args);

        reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/kps_project_fused/device/kernels/dataflow/reader_grid_precompute.cpp",
            all_cores,
            ReaderDataMovementConfig(precompute_ct_args));

        std::vector<uint32_t> writer_ct = {out_cb_index, out_page_size, nc * NL, NL};
        TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);
        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/kps_project_fused/device/kernels/dataflow/writer_kps_project.cpp",
            all_cores,
            WriterDataMovementConfig(writer_ct));

        uint32_t anchor_offset = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t ca = core_group_1.contains(core) ? n_anchors_1 : n_anchors_2;
            SetRuntimeArgs(program, reader_kernel_id, core, {
                t.key_points.buffer()->address(), ca, anchor_offset, n});
            SetRuntimeArgs(program, writer_kernel_id, core, {
                output_tensor.buffer()->address(), ca, anchor_offset, n});
            anchor_offset += ca;
        }
    } else if (fused_precompute) {
        // ===== Fused precompute: full kps_project + multi-level precompute in one pass =====
        // Input: key_points [n, pts, 3], anchor [n, 1, 11], proj [nc, 4, 4], wh [nc, 1, 2]
        // Output: [nc*NL, N, 1, pts*6] bf16 precomputed grid (all levels, camera-major)
        std::vector<uint32_t> reader_ct_args = {
            kp_cb_index, anchor_cb_index, proj_cb_index, wh_cb_index, out_cb_index,
            nc, num_pts, kp_page_size, anchor_page_size, proj_page_size, wh_page_size, out_page_size,
        };
        TensorAccessorArgs(*t.key_points.buffer()).append_to(reader_ct_args);
        TensorAccessorArgs(*t.anchor.buffer()).append_to(reader_ct_args);
        TensorAccessorArgs(*t.projection_mat.buffer()).append_to(reader_ct_args);
        TensorAccessorArgs(*t.image_wh.buffer()).append_to(reader_ct_args);
        // Fused precompute: PRECOMPUTE=1, NL from spatial_shapes
        // H,W passed via runtime args to avoid program cache collision
        reader_ct_args.push_back(1U);  // PRECOMPUTE=1
        reader_ct_args.push_back(NL);  // NL=4
        for (uint32_t i = 0; i < 4; i++) {
            reader_ct_args.push_back(0);  // placeholder H (runtime)
            reader_ct_args.push_back(0);  // placeholder W (runtime)
        }

        reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/kps_project_fused/device/kernels/dataflow/reader_kps_project.cpp",
            all_cores,
            ReaderDataMovementConfig(reader_ct_args));

        // Writer: NC = nc * NL (all levels)
        std::vector<uint32_t> writer_ct = {out_cb_index, out_page_size, nc * NL, NL};
        TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);
        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/kps_project_fused/device/kernels/dataflow/writer_kps_project.cpp",
            all_cores,
            WriterDataMovementConfig(writer_ct));

        uint32_t anchor_offset = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t ca = core_group_1.contains(core) ? n_anchors_1 : n_anchors_2;
            // Runtime args: addresses + work split + spatial shapes (H0,W0,H1,W1,H2,W2,H3,W3)
            std::vector<uint32_t> reader_rt = {
                t.key_points.buffer()->address(), t.anchor.buffer()->address(),
                t.projection_mat.buffer()->address(), t.image_wh.buffer()->address(),
                ca, anchor_offset, n,
            };
            for (auto& [h, w] : attrs.spatial_shapes) {
                reader_rt.push_back(h);
                reader_rt.push_back(w);
            }
            for (uint32_t j = attrs.spatial_shapes.size(); j < 4; j++) {
                reader_rt.push_back(0);
                reader_rt.push_back(0);
            }
            SetRuntimeArgs(program, reader_kernel_id, core, reader_rt);
            SetRuntimeArgs(program, writer_kernel_id, core, {
                output_tensor.buffer()->address(), ca, anchor_offset, n});
            anchor_offset += ca;
        }
    } else {
        // ===== Standard mode: full KPS projection =====
        std::vector<uint32_t> reader_ct_args = {
            kp_cb_index, anchor_cb_index, proj_cb_index, wh_cb_index, out_cb_index,
            nc, num_pts, kp_page_size, anchor_page_size, proj_page_size, wh_page_size, out_page_size,
        };
        TensorAccessorArgs(*t.key_points.buffer()).append_to(reader_ct_args);
        TensorAccessorArgs(*t.anchor.buffer()).append_to(reader_ct_args);
        TensorAccessorArgs(*t.projection_mat.buffer()).append_to(reader_ct_args);
        TensorAccessorArgs(*t.image_wh.buffer()).append_to(reader_ct_args);
        // Standard mode: no precompute args needed
        reader_ct_args.push_back(0U);  // PRECOMPUTE=0
        reader_ct_args.push_back(0U);  // NL=0

        reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/kps_project_fused/device/kernels/dataflow/reader_kps_project.cpp",
            all_cores,
            ReaderDataMovementConfig(reader_ct_args));

        std::vector<uint32_t> writer_ct = {out_cb_index, out_page_size, nc, 0U};
        TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_ct);
        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/pool/kps_project_fused/device/kernels/dataflow/writer_kps_project.cpp",
            all_cores,
            WriterDataMovementConfig(writer_ct));

        uint32_t anchor_offset = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord& core = logical_cores[i];
            const uint32_t ca = core_group_1.contains(core) ? n_anchors_1 : n_anchors_2;
            SetRuntimeArgs(program, reader_kernel_id, core, {
                t.key_points.buffer()->address(), t.anchor.buffer()->address(),
                t.projection_mat.buffer()->address(), t.image_wh.buffer()->address(),
                ca, anchor_offset, n, 0U, 0U});  // rt_H=0, rt_W=0 (unused in standard mode)
            SetRuntimeArgs(program, writer_kernel_id, core, {
                output_tensor.buffer()->address(), ca, anchor_offset, n});
            anchor_offset += ca;
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .num_cores = num_cores,
            .logical_cores = logical_cores,
            .precompute_mode = attrs.precompute_grid,
            .fused_precompute_mode = fused_precompute,
        }};
}

void KpsProjectFusedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const KpsProjectFusedParams& attrs,
    const KpsProjectFusedInputs& t,
    Tensor& output_tensor) {
    auto& prog = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    for (uint32_t i = 0; i < sv.num_cores; i++) {
        auto& r = GetRuntimeArgs(prog, sv.reader_kernel_id, sv.logical_cores[i]);
        r[0] = t.key_points.buffer()->address();
        if (!sv.precompute_mode || sv.fused_precompute_mode) {
            r[1] = t.anchor.buffer()->address();
            r[2] = t.projection_mat.buffer()->address();
            r[3] = t.image_wh.buffer()->address();
            // Update H, W for fused precompute (runtime args 7+)
            if (sv.fused_precompute_mode) {
                for (uint32_t j = 0; j < attrs.spatial_shapes.size() && j < 4; j++) {
                    r[7 + j * 2] = attrs.spatial_shapes[j].first;
                    r[7 + j * 2 + 1] = attrs.spatial_shapes[j].second;
                }
            }
        }
        GetRuntimeArgs(prog, sv.writer_kernel_id, sv.logical_cores[i])[0] =
            output_tensor.buffer()->address();
    }
}

}  // namespace ttnn::prim
