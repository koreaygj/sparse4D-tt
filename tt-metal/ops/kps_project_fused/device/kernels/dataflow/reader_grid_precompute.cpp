// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Lightweight grid precompute reader:
// Input: normalized grid [nc, N, 1, pts*2] f32 (from kps_project_fused standard output)
// Output: per-level precomputed grid [nc*NL, N, 1, pts*6] bf16
// Each output page = [pts*6] bf16 = [h0_bits, w0_bits, w_nw, w_ne, w_sw, w_se] × pts

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = (bits & 0x10000u) ? 0x8000u : 0x7FFFu;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

void kernel_main() {
    uint32_t grid_addr      = get_arg_val<uint32_t>(0);
    uint32_t num_anchors    = get_arg_val<uint32_t>(1);
    uint32_t anchor_offset  = get_arg_val<uint32_t>(2);
    uint32_t total_anchors  = get_arg_val<uint32_t>(3);

    constexpr uint32_t grid_cb_index  = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_index   = get_compile_time_arg_val(1);
    constexpr uint32_t NC             = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_PTS        = get_compile_time_arg_val(3);
    constexpr uint32_t grid_page_size = get_compile_time_arg_val(4);  // pts*2*sizeof(float), aligned
    constexpr uint32_t out_page_size  = get_compile_time_arg_val(5);  // pts*6*sizeof(uint16_t), aligned
    constexpr uint32_t NL             = get_compile_time_arg_val(6);

    // Spatial shapes from compile-time args
    constexpr uint32_t SHAPES_OFFSET = 7;
    // Read H, W arrays from kernel_compile_time_args at runtime
    // (can't use get_compile_time_arg_val in a loop with non-constexpr index)

    constexpr auto grid_ta = TensorAccessorArgs<SHAPES_OFFSET + 8>();  // after 4 pairs of H,W
    const auto grid_acc = TensorAccessor(grid_ta, grid_addr, grid_page_size);

    // Pre-read spatial shapes into local array
    uint32_t H[4], W[4];
    for (uint32_t i = 0; i < NL && i < 4; i++) {
        H[i] = kernel_compile_time_args[SHAPES_OFFSET + i * 2];
        W[i] = kernel_compile_time_args[SHAPES_OFFSET + i * 2 + 1];
    }

    for (uint32_t a_idx = 0; a_idx < num_anchors; a_idx++) {
        uint32_t a = anchor_offset + a_idx;

        for (uint32_t c = 0; c < NC; c++) {
            // Read grid page: [pts*2] f32 = normalized (grid_x, grid_y) per point
            uint32_t grid_page_id = c * total_anchors + a;
            uint32_t grid_l1 = get_write_ptr(grid_cb_index);
            noc_async_read(grid_acc.get_noc_addr(grid_page_id), grid_l1, grid_page_size);
            noc_async_read_barrier();

            volatile tt_l1_ptr float* gf = reinterpret_cast<volatile tt_l1_ptr float*>(grid_l1);

            // Reserve NL output pages for this (camera, anchor)
            cb_reserve_back(out_cb_index, NL);
            uint32_t out_base = get_write_ptr(out_cb_index);

            for (uint32_t p = 0; p < NUM_PTS; p++) {
                float grid_x = gf[p * 2 + 0];
                float grid_y = gf[p * 2 + 1];

                for (uint32_t lvl = 0; lvl < NL; lvl++) {
                    float hs = (float)H[lvl] * 0.5f;
                    float ws = (float)W[lvl] * 0.5f;
                    float h_img = (grid_y + 1.0f) * hs - 0.5f;
                    float w_img = (grid_x + 1.0f) * ws - 0.5f;

                    int32_t h0 = (int32_t)__builtin_floorf(h_img);
                    int32_t w0 = (int32_t)__builtin_floorf(w_img);
                    int32_t h1 = h0 + 1;
                    int32_t w1 = w0 + 1;
                    float dy = h_img - (float)h0;
                    float dx = w_img - (float)w0;

                    bool h0v = (h0 >= 0) && (h0 < (int32_t)H[lvl]);
                    bool h1v = (h1 >= 0) && (h1 < (int32_t)H[lvl]);
                    bool w0v = (w0 >= 0) && (w0 < (int32_t)W[lvl]);
                    bool w1v = (w1 >= 0) && (w1 < (int32_t)W[lvl]);

                    float wt_nw = (h0v && w0v) ? (1.0f - dy) * (1.0f - dx) : 0.0f;
                    float wt_ne = (h0v && w1v) ? (1.0f - dy) * dx : 0.0f;
                    float wt_sw = (h1v && w0v) ? dy * (1.0f - dx) : 0.0f;
                    float wt_se = (h1v && w1v) ? dy * dx : 0.0f;

                    volatile tt_l1_ptr uint16_t* out_bf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        out_base + lvl * out_page_size);
                    int16_t h0_i16 = (int16_t)h0;
                    int16_t w0_i16 = (int16_t)w0;
                    out_bf[p * 6 + 0] = *reinterpret_cast<uint16_t*>(&h0_i16);
                    out_bf[p * 6 + 1] = *reinterpret_cast<uint16_t*>(&w0_i16);
                    out_bf[p * 6 + 2] = f32_to_bf16_rne(wt_nw);
                    out_bf[p * 6 + 3] = f32_to_bf16_rne(wt_ne);
                    out_bf[p * 6 + 4] = f32_to_bf16_rne(wt_sw);
                    out_bf[p * 6 + 5] = f32_to_bf16_rne(wt_se);
                }
            }

            cb_push_back(out_cb_index, NL);
        }
    }
}
