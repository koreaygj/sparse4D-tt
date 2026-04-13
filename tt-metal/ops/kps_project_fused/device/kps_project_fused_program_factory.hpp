// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/device_operation.hpp"
#include "kps_project_fused_device_operation_types.hpp"

namespace ttnn::prim {

struct KpsProjectFusedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        uint32_t num_cores;
        std::vector<tt::tt_metal::CoreCoord> logical_cores;
        bool precompute_mode = false;
        bool fused_precompute_mode = false;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const KpsProjectFusedParams& params,
        const KpsProjectFusedInputs& inputs,
        tt::tt_metal::Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const KpsProjectFusedParams& params,
        const KpsProjectFusedInputs& inputs,
        tt::tt_metal::Tensor& output);
};

}  // namespace ttnn::prim
