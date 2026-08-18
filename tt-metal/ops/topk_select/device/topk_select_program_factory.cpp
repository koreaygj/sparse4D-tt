// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "topk_select_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

// One core, one dataflow kernel, no compute. The whole job is ~5k integer L1
// operations — well under the dispatch floor — so parallelism has nothing to
// recover here; what mattered was not sorting 31 rows of tile padding.
TopkSelectProgramFactory::cached_program_t TopkSelectProgramFactory::create(
    const TopkSelectParams& attrs,
    const TopkSelectInputs& t,
    Tensor& /*output_tensor*/) {

    Program program{};
    const CoreCoord core{0, 0};
    const CoreRangeSet core_range{CoreRange{core, core}};

    const uint32_t nt = (attrs.n + 31) / 32;
    const uint32_t scores_bytes = nt * 64;                       // row 0, 64 B per tile
    const uint32_t rec_bytes = ((2 * attrs.n * 4 + 31) / 32) * 32;  // ping-pong records
    const uint32_t cnt_bytes = 256 * 4;
    const uint32_t val_bytes = ((attrs.k * 2 + 31) / 32) * 32;
    const uint32_t idx_bytes = ((attrs.k * 4 + 31) / 32) * 32;

    constexpr uint32_t SCORES_CB = tt::CBIndex::c_0;
    constexpr uint32_t REC_CB    = tt::CBIndex::c_1;
    constexpr uint32_t CNT_CB    = tt::CBIndex::c_2;
    constexpr uint32_t VAL_CB    = tt::CBIndex::c_3;
    constexpr uint32_t IDX_CB    = tt::CBIndex::c_4;

    auto mk = [&](uint32_t cb, uint32_t bytes, DataFormat fmt) {
        auto cfg = CircularBufferConfig(bytes, {{cb, fmt}}).set_page_size(cb, bytes);
        CreateCircularBuffer(program, core_range, cfg);
    };
    mk(SCORES_CB, scores_bytes, DataFormat::UInt16);
    mk(REC_CB, rec_bytes, DataFormat::UInt32);
    mk(CNT_CB, cnt_bytes, DataFormat::UInt32);
    mk(VAL_CB, val_bytes, DataFormat::UInt16);
    mk(IDX_CB, idx_bytes, DataFormat::UInt32);

    std::vector<uint32_t> ct_args = {
        attrs.n,    // 0
        attrs.k,    // 1
        SCORES_CB,  // 2
        REC_CB,     // 3
        CNT_CB,     // 4
        VAL_CB,     // 5
        IDX_CB,     // 6
    };
    TensorAccessorArgs(*t.scores.buffer()).append_to(ct_args);
    TensorAccessorArgs(*t.values.buffer()).append_to(ct_args);
    TensorAccessorArgs(*t.indices.buffer()).append_to(ct_args);

    KernelHandle kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/topk_select/device/kernels/dataflow/topk_select.cpp",
        core_range,
        WriterDataMovementConfig(ct_args));

    SetRuntimeArgs(program, kernel_id, core, {
        t.scores.buffer()->address(),
        t.values.buffer()->address(),
        t.indices.buffer()->address()});

    return cached_program_t{
        std::move(program), shared_variables_t{.kernel_id = kernel_id}};
}

void TopkSelectProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TopkSelectParams&,
    const TopkSelectInputs& t,
    Tensor& /*output_tensor*/) {
    auto& prog = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    auto& r = GetRuntimeArgs(prog, sv.kernel_id, CoreCoord{0, 0});
    r[0] = t.scores.buffer()->address();
    r[1] = t.values.buffer()->address();
    r[2] = t.indices.buffer()->address();
}

}  // namespace ttnn::prim
