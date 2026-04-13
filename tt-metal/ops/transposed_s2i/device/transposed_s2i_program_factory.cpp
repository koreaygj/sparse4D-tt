// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "transposed_s2i_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

TransposedS2iProgramFactory::cached_program_t TransposedS2iProgramFactory::create(
    const TransposedS2iParams& attrs,
    const TransposedS2iInputs& t,
    Tensor& output_tensor) {

    Program program{};

    const uint32_t NC = attrs.num_cams;
    const uint32_t K = attrs.num_pts;
    const uint32_t N = attrs.num_anchors;
    const uint32_t C = t.input.logical_shape()[-1];
    const uint32_t stick_size = C * sizeof(uint16_t);

    // Input is L1 HEIGHT_SHARDED. Get shard info.
    const auto& shard_spec_val = t.input.memory_config().shard_spec().value();
    const auto& shard_shape = shard_spec_val.shape;
    const uint32_t shard_h = shard_shape[0];  // sticks per core

    // Use the same core range as the input shard
    const auto& core_range = shard_spec_val.grid;
    auto logical_cores = corerange_to_cores(core_range, std::nullopt, true);
    const uint32_t num_cores = logical_cores.size();

    const uint32_t total_sticks = NC * N * K;

    // Compile-time args
    std::vector<uint32_t> ct_args = {
        stick_size,         // 0
        N,                  // 1
        K,                  // 2
        NC,                 // 3
        attrs.num_levels,   // 4: NL
        attrs.level,        // 5: LEVEL
    };
    TensorAccessorArgs(*output_tensor.buffer()).append_to(ct_args);

    KernelHandle kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/transposed_s2i/device/kernels/dataflow/transposed_s2i.cpp",
        core_range,
        WriterDataMovementConfig(ct_args));

    // Runtime args per core
    const auto& in_buffer = *t.input.buffer();
    uint32_t stick_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord& core = logical_cores[i];
        uint32_t sticks_this_core = std::min(shard_h, total_sticks - stick_offset);
        if (stick_offset >= total_sticks) sticks_this_core = 0;

        SetRuntimeArgs(program, kernel_id, core, {
            output_tensor.buffer()->address(),
            sticks_this_core,
            stick_offset,
            in_buffer.address()});
        stick_offset += shard_h;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .kernel_id = kernel_id,
            .num_cores = num_cores,
            .logical_cores = logical_cores,
        }};
}

void TransposedS2iProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TransposedS2iParams&,
    const TransposedS2iInputs& t,
    Tensor& output_tensor) {
    auto& prog = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    for (uint32_t i = 0; i < sv.num_cores; i++) {
        auto& r = GetRuntimeArgs(prog, sv.kernel_id, sv.logical_cores[i]);
        r[0] = output_tensor.buffer()->address();
        r[3] = t.input.buffer()->address();
    }
}

}  // namespace ttnn::prim
