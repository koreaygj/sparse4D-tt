// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/device_operation.hpp"
#include "grid_compact_device_operation_types.hpp"

namespace ttnn::prim {

struct GridCompactProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle kernel_id;
        // Every core runs the bounds test and so holds the grid address; cores[0] is the
        // one that also does the writes.
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GridCompactParams& params,
        const GridCompactInputs& inputs,
        tt::tt_metal::Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GridCompactParams& params,
        const GridCompactInputs& inputs,
        tt::tt_metal::Tensor& output);
};

}  // namespace ttnn::prim
