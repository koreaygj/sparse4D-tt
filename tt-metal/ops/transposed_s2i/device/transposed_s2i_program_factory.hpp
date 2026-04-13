// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/device_operation.hpp"
#include "transposed_s2i_device_operation_types.hpp"

namespace ttnn::prim {

struct TransposedS2iProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle kernel_id;
        uint32_t num_cores;
        std::vector<tt::tt_metal::CoreCoord> logical_cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const TransposedS2iParams& params,
        const TransposedS2iInputs& inputs,
        tt::tt_metal::Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TransposedS2iParams& params,
        const TransposedS2iInputs& inputs,
        tt::tt_metal::Tensor& output);
};

}  // namespace ttnn::prim
