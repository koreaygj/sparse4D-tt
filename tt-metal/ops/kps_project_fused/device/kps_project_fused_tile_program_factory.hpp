// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "kps_project_fused_device_operation_types.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::prim {

// 3-kernel TILE-based program factory for kps_project_fused
// Uses matmul_tiles for rotation and projection (FPU), scalar for perspective divide
struct KpsProjectFusedTileProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        uint32_t num_cores;
        std::vector<CoreCoord> logical_cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const KpsProjectFusedParams& attrs,
        const KpsProjectFusedInputs& t,
        tt::tt_metal::Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const KpsProjectFusedParams& attrs,
        const KpsProjectFusedInputs& t,
        tt::tt_metal::Tensor& output_tensor);
};

}  // namespace ttnn::prim
