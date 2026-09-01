// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include <tt-metalium/host_api.hpp>
#include "ttnn/device_operation.hpp"
#include "grid_precompute_device_operation_types.hpp"

namespace ttnn::prim {

struct GridPrecomputeProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_id;
        tt::tt_metal::KernelHandle writer_id;
        std::vector<tt::tt_metal::CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GridPrecomputeParams& attrs,
        const GridPrecomputeInputs& t,
        tt::tt_metal::Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GridPrecomputeParams& attrs,
        const GridPrecomputeInputs& t,
        tt::tt_metal::Tensor& output_tensor);
};

}  // namespace ttnn::prim
