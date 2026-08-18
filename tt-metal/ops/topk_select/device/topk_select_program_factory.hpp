// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "topk_select_device_operation_types.hpp"

namespace ttnn::prim {

struct TopkSelectProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle kernel_id;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const TopkSelectParams& attrs,
        const TopkSelectInputs& t,
        tt::tt_metal::Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TopkSelectParams& attrs,
        const TopkSelectInputs& t,
        tt::tt_metal::Tensor& output_tensor);
};

}  // namespace ttnn::prim
