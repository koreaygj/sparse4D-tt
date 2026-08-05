// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Compacts a DFA sampling grid down to the rows that actually hit an image.
// ~80% of (camera, anchor) rows project entirely outside their camera, and
// grid_sample pays full pipeline cost for them even though it skips the reads.
struct GridCompactParams {
    uint32_t num_rows;    // nc * N, e.g. 2700
    uint32_t num_pts;     // K, points per row
    uint32_t row_width;   // padded floats per row (>= 2*K + 1, camera id goes at 2*K)
    uint32_t capacity;    // fixed rows kept; shapes must not vary or ttnn recompiles
    uint32_t anchors;     // N, so camera id = row / N
    uint32_t thr_x_bits;  // fp32 bit pattern of the in-bounds threshold, x axis
    uint32_t thr_y_bits;  // ... and y. They differ a lot: the bound is 1 + 1/W on x
                          // and 1 + 1/H on y, and the coarsest level is 8 x 22.
    uint32_t flag_width;  // padded anchors per camera in the flags tensor
    uint32_t bidx_width;  // row width of the batch-index tensor (0 = per-camera mode)
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct GridCompactInputs {
    const tt::tt_metal::Tensor& grid;   // DRAM [nc, N, 1, row_width] fp32
    const tt::tt_metal::Tensor& cgrid;  // DRAM [1, capacity, 1, row_width] fp32, preallocated
    const tt::tt_metal::Tensor& index;  // DRAM [1, 1, 1, nc*capacity] uint32, preallocated
    const tt::tt_metal::Tensor& flags;  // DRAM [nc, 1, 1, flag_width] bf16, preallocated
    // Pooled mode. Without it the kept rows are written as one fixed block per camera,
    // because grid_sample derives a row's camera from its POSITION — which means CAP has
    // to cover the busiest camera, and the cameras' busy frames do not coincide (measured:
    // 3 x 563 = 1689 rows per-camera versus 902 pooled, for the same zero-loss guarantee).
    // With bidx the rows go into one shared list and each carries its camera, which
    // grid_sample now accepts via its batch_index argument.
    std::optional<tt::tt_metal::Tensor> bidx;  // DRAM [1, 1, capacity, bidx_width] uint32
};

}  // namespace ttnn::prim
