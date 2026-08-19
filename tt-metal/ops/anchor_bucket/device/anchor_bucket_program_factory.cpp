// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "anchor_bucket_program_factory.hpp"

namespace ttnn::prim {
using namespace tt;
using namespace tt::tt_metal;

AnchorBucketProgramFactory::cached_program_t AnchorBucketProgramFactory::create(
    const AnchorBucketParams& attrs,
    const AnchorBucketInputs& t,
    Tensor& /*output_tensor*/) {

    Program program{};
    const CoreCoord core{0, 0};
    const CoreRangeSet core_range{CoreRange{core, core}};

    const uint32_t nc = t.flags.logical_shape()[0];
    const uint32_t fw = t.flags.logical_shape()[-1];
    const uint32_t npad = t.perm.logical_shape()[-1];

    constexpr uint32_t CB_FLG = tt::CBIndex::c_0;
    constexpr uint32_t CB_OUT = tt::CBIndex::c_1;
    const uint32_t flg_bytes = nc * (((fw * 2 + 31) / 32) * 32);
    const uint32_t out_bytes = ((npad * 8 + 32 * 4 + npad + 31) / 32) * 32;

    auto mk = [&](uint32_t cb, uint32_t bytes, DataFormat fmt) {
        auto cfg = CircularBufferConfig(bytes, {{cb, fmt}}).set_page_size(cb, bytes);
        CreateCircularBuffer(program, core_range, cfg);
    };
    mk(CB_FLG, flg_bytes, DataFormat::UInt16);
    mk(CB_OUT, out_bytes, DataFormat::UInt32);

    // RM sticks are DRAM-aligned: the buffer's page stride is the ALIGNED
    // size (900*2=1800 -> 1824), so the accessor must use it, not fw*2.
    const uint32_t fpage = t.flags.buffer()->aligned_page_size();
    std::vector<uint32_t> ct = {attrs.num_anchors, nc, fw, npad, CB_FLG, CB_OUT, fpage};
    TensorAccessorArgs(*t.flags.buffer()).append_to(ct);
    TensorAccessorArgs(*t.perm.buffer()).append_to(ct);
    TensorAccessorArgs(*t.inv.buffer()).append_to(ct);
    TensorAccessorArgs(*t.live.buffer()).append_to(ct);

    KernelHandle kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/pool/anchor_bucket/device/kernels/dataflow/anchor_bucket.cpp",
        core_range, WriterDataMovementConfig(ct));

    SetRuntimeArgs(program, kernel_id, core, {
        t.flags.buffer()->address(), t.perm.buffer()->address(),
        t.inv.buffer()->address(), t.live.buffer()->address()});

    return cached_program_t{
        std::move(program), shared_variables_t{.kernel_id = kernel_id}};
}

void AnchorBucketProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const AnchorBucketParams&,
    const AnchorBucketInputs& t,
    Tensor& /*output_tensor*/) {
    auto& r = GetRuntimeArgs(cached_program.program,
                             cached_program.shared_variables.kernel_id,
                             CoreCoord{0, 0});
    r[0] = t.flags.buffer()->address();
    r[1] = t.perm.buffer()->address();
    r[2] = t.inv.buffer()->address();
    r[3] = t.live.buffer()->address();
}

}  // namespace ttnn::prim
