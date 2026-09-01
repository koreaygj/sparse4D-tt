// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Converts the compacted Q14 sampling grid into grid_sample's precomputed
// 6-field form (h0, w0, four bilinear weights), one output tensor per FPN
// level, with all arithmetic on the Tensix engines.
//
// Exists because grid_sample's reader otherwise derives those six values per
// point per level in soft float on an FPU-less core — measured at ~62% of the
// op, and the precomputed path measured 2.78x. Runs AFTER compaction on
// purpose: the kept rows are ~2.3x fewer than the full grid, and the variant
// that ran this math before compaction was priced at +38.5 ms/frame.
struct GridPrecomputeParams {
    uint32_t num_pts;      // K, points per row
    uint32_t num_levels;   // FPN levels = number of output tensors
    uint32_t num_rows;     // capacity of the compacted grid
    uint32_t row_width;    // values per cgrid row (>= 2*K)
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct GridPrecomputeInputs {
    const tt::tt_metal::Tensor& cgrid;    // DRAM [1, cap, 1, row_width] UINT16 (Q14)
    // 23 f32 TILE-layout tiles built once in python: per-level affine D and
    // bound C, plus the level-independent selector matrices. See the reader
    // for the ordering contract.
    const tt::tt_metal::Tensor& consts;
    // Preallocated DRAM [1, cap, 1, K*6] BFLOAT16 ROW_MAJOR, one per level.
    // Row r corresponds 1:1 to cgrid row r, so bidx / index / flags from
    // grid_compact apply unchanged.
    const tt::tt_metal::Tensor& out0;
    const tt::tt_metal::Tensor& out1;
    const tt::tt_metal::Tensor& out2;
    const tt::tt_metal::Tensor& out3;
};

}  // namespace ttnn::prim
