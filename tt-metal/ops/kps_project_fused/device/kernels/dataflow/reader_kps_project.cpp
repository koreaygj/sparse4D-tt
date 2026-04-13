// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Fused KPS Rotation + Translation + Projection + Normalize
//
// Per anchor:
//   1. Read key_points[num_pts][3] (bf16) and anchor[11] (bf16)
//   2. Rotate key_points by yaw (cos/sin from anchor)
//   3. Translate by center (x,y,z from anchor)
//   4. For each camera: project via 4x4 matrix (f32), perspective divide, normalize
//   5. Write output[cam][anchor][1][num_pts*2] (f32) to output CB
// =============================================================================

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t ANC_X = 0, ANC_Y = 1, ANC_Z = 2;
constexpr uint32_t ANC_SIN_YAW = 6, ANC_COS_YAW = 7;

// bf16 → f32 conversion: bf16 is upper 16 bits of f32
inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

// f32 → bf16 with round-to-nearest-even (matching TT TILE hardware and PyTorch bf16)
inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(float));
    // Round-to-nearest-even: add 0x7FFF + bit[16] (the "sticky" rounding)
    uint32_t rounding_bias = (bits & 0x10000u) ? 0x8000u : 0x7FFFu;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

// Truncate f32 to bf16 precision and back — matches TILE bf16 multiply output
inline float bf16_trunc(float f) {
    return bf16_to_f32(f32_to_bf16_rne(f));
}

void kernel_main() {
    uint32_t kp_addr       = get_arg_val<uint32_t>(0);
    uint32_t anchor_addr   = get_arg_val<uint32_t>(1);
    uint32_t proj_addr     = get_arg_val<uint32_t>(2);
    uint32_t wh_addr       = get_arg_val<uint32_t>(3);
    uint32_t num_anchors   = get_arg_val<uint32_t>(4);
    uint32_t anchor_offset = get_arg_val<uint32_t>(5);
    uint32_t total_anchors = get_arg_val<uint32_t>(6);
    // Runtime spatial shapes for precompute (up to 4 levels)
    uint32_t rt_H[4], rt_W[4];
    for (uint32_t i = 0; i < 4; i++) {
        rt_H[i] = get_arg_val<uint32_t>(7 + i * 2);
        rt_W[i] = get_arg_val<uint32_t>(8 + i * 2);
    }

    constexpr uint32_t kp_cb_index     = get_compile_time_arg_val(0);
    constexpr uint32_t anchor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t proj_cb_index   = get_compile_time_arg_val(2);
    constexpr uint32_t wh_cb_index     = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb_index    = get_compile_time_arg_val(4);
    constexpr uint32_t NC              = get_compile_time_arg_val(5);
    constexpr uint32_t NUM_PTS         = get_compile_time_arg_val(6);
    constexpr uint32_t kp_page_size    = get_compile_time_arg_val(7);
    constexpr uint32_t anchor_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t proj_page_size  = get_compile_time_arg_val(9);
    constexpr uint32_t wh_page_size    = get_compile_time_arg_val(10);
    constexpr uint32_t out_page_size   = get_compile_time_arg_val(11);

    constexpr auto kp_args = TensorAccessorArgs<12>();
    constexpr auto anchor_args = TensorAccessorArgs<kp_args.next_compile_time_args_offset()>();
    constexpr auto proj_args = TensorAccessorArgs<anchor_args.next_compile_time_args_offset()>();
    constexpr auto wh_args = TensorAccessorArgs<proj_args.next_compile_time_args_offset()>();

    // Precompute grid args (after tensor accessors)
    constexpr uint32_t PRECOMPUTE_OFFSET = wh_args.next_compile_time_args_offset();
    constexpr uint32_t PRECOMPUTE = get_compile_time_arg_val(PRECOMPUTE_OFFSET);
    constexpr uint32_t NL = get_compile_time_arg_val(PRECOMPUTE_OFFSET + 1);

    const auto kp_accessor = TensorAccessor(kp_args, kp_addr, kp_page_size);
    const auto anchor_accessor = TensorAccessor(anchor_args, anchor_addr, anchor_page_size);
    const auto proj_accessor = TensorAccessor(proj_args, proj_addr, proj_page_size);
    const auto wh_accessor = TensorAccessor(wh_args, wh_addr, wh_page_size);

    // Load projection matrices [nc, 4, 4] f32 into L1
    // Each row (4 floats = 16 bytes) is one page, aligned to proj_page_size
    uint32_t proj_l1_addr = get_write_ptr(proj_cb_index);
    for (uint32_t page = 0; page < NC * 4; page++) {
        uint64_t noc_addr = proj_accessor.get_noc_addr(page);
        noc_async_read(noc_addr, proj_l1_addr + page * proj_page_size, proj_page_size);
    }
    noc_async_read_barrier();
    // proj_page_size may be > 16 bytes due to alignment.
    // Access via page stride, not element stride.
    constexpr uint32_t proj_page_stride = proj_page_size / sizeof(float);  // floats per page (including padding)

    // Load image_wh [nc, 1, 2] f32 into L1
    uint32_t wh_l1_addr = get_write_ptr(wh_cb_index);
    for (uint32_t page = 0; page < NC; page++) {
        uint64_t noc_addr = wh_accessor.get_noc_addr(page);
        noc_async_read(noc_addr, wh_l1_addr + page * wh_page_size, wh_page_size);
    }
    noc_async_read_barrier();
    constexpr uint32_t wh_page_stride = wh_page_size / sizeof(float);

    for (uint32_t a_idx = 0; a_idx < num_anchors; a_idx++) {
        uint32_t a = anchor_offset + a_idx;

        // Read anchor[a]
        uint32_t anchor_l1 = get_write_ptr(anchor_cb_index);
        {
            uint64_t noc_addr = anchor_accessor.get_noc_addr(a);
            noc_async_read(noc_addr, anchor_l1, anchor_page_size);
            noc_async_read_barrier();
        }
        volatile tt_l1_ptr uint16_t* anc_bf16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(anchor_l1);
        float cos_yaw = bf16_to_f32(anc_bf16[ANC_COS_YAW]);
        float sin_yaw = bf16_to_f32(anc_bf16[ANC_SIN_YAW]);
        float cx = bf16_to_f32(anc_bf16[ANC_X]);
        float cy = bf16_to_f32(anc_bf16[ANC_Y]);
        float cz = bf16_to_f32(anc_bf16[ANC_Z]);

        // Read key_points[a, :num_pts, :3]
        uint32_t kp_l1 = get_write_ptr(kp_cb_index);
        uint32_t kp_start_page = a * NUM_PTS;
        for (uint32_t p = 0; p < NUM_PTS; p++) {
            uint64_t noc_addr = kp_accessor.get_noc_addr(kp_start_page + p);
            noc_async_read(noc_addr, kp_l1 + p * kp_page_size, kp_page_size);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* kp_bf16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(kp_l1);

        for (uint32_t c = 0; c < NC; c++) {

            // Load projection matrix for camera c
            volatile tt_l1_ptr float* proj_base = reinterpret_cast<volatile tt_l1_ptr float*>(proj_l1_addr);
            uint32_t p_base = c * 4 * proj_page_stride;
            float P0  = proj_base[p_base + 0*proj_page_stride + 0];
            float P1  = proj_base[p_base + 0*proj_page_stride + 1];
            float P2  = proj_base[p_base + 0*proj_page_stride + 2];
            float P3  = proj_base[p_base + 0*proj_page_stride + 3];
            float P4  = proj_base[p_base + 1*proj_page_stride + 0];
            float P5  = proj_base[p_base + 1*proj_page_stride + 1];
            float P6  = proj_base[p_base + 1*proj_page_stride + 2];
            float P7  = proj_base[p_base + 1*proj_page_stride + 3];
            float P8  = proj_base[p_base + 2*proj_page_stride + 0];
            float P9  = proj_base[p_base + 2*proj_page_stride + 1];
            float P10 = proj_base[p_base + 2*proj_page_stride + 2];
            float P11 = proj_base[p_base + 2*proj_page_stride + 3];

            volatile tt_l1_ptr float* wh_base = reinterpret_cast<volatile tt_l1_ptr float*>(wh_l1_addr);
            float inv_w = 1.0f / wh_base[c * wh_page_stride + 0];
            float inv_h = 1.0f / wh_base[c * wh_page_stride + 1];

            if constexpr (PRECOMPUTE) {
                // ===== Precomputed grid mode: output per-level [h0, w0, w_nw, w_ne, w_sw, w_se] bf16 =====
                // Buffer normalized grid coords per point for this camera
                float buf_gx[13], buf_gy[13];  // max 13 pts

                // Load spatial shapes once
                uint32_t H_arr[4], W_arr[4];
                for (uint32_t i = 0; i < NL && i < 4; i++) {
                    H_arr[i] = kernel_compile_time_args[PRECOMPUTE_OFFSET + 2 + i * 2];
                    W_arr[i] = kernel_compile_time_args[PRECOMPUTE_OFFSET + 2 + i * 2 + 1];
                }

                for (uint32_t p = 0; p < NUM_PTS; p++) {
                    uint32_t bf16_offset = p * (kp_page_size / sizeof(uint16_t));
                    float kx = bf16_to_f32(kp_bf16[bf16_offset + 0]);
                    float ky = bf16_to_f32(kp_bf16[bf16_offset + 1]);
                    float kz = bf16_to_f32(kp_bf16[bf16_offset + 2]);

                    float cos_kx = bf16_trunc(cos_yaw * kx);
                    float sin_ky = bf16_trunc(sin_yaw * ky);
                    float rx = bf16_trunc(cos_kx - sin_ky);
                    float sin_kx = bf16_trunc(sin_yaw * kx);
                    float cos_ky = bf16_trunc(cos_yaw * ky);
                    float ry = bf16_trunc(sin_kx + cos_ky);
                    float rz = kz;
                    float px = bf16_trunc(rx + cx);
                    float py = bf16_trunc(ry + cy);
                    float pz = bf16_trunc(rz + cz);

                    float proj_x = px * P0 + py * P1 + pz * P2 + P3;
                    float proj_y = px * P4 + py * P5 + pz * P6 + P7;
                    float proj_z = px * P8 + py * P9 + pz * P10 + P11;
                    float z_safe = proj_z > 1e-5f ? proj_z : 1e-5f;
                    float inv_z = 1.0f / z_safe;
                    float ndx = proj_x * inv_z;
                    float ndy = proj_y * inv_z;
                    float grid_x = ndx * inv_w * 2.0f - 1.0f;
                    float grid_y = ndy * inv_h * 2.0f - 1.0f;
                    if (grid_x < -2.0f) grid_x = -2.0f;
                    if (grid_x >  2.0f) grid_x =  2.0f;
                    if (grid_y < -2.0f) grid_y = -2.0f;
                    if (grid_y >  2.0f) grid_y =  2.0f;

                    buf_gx[p] = grid_x;
                    buf_gy[p] = grid_y;
                }

                // Multi-level: batch reserve NL pages per camera
                cb_reserve_back(out_cb_index, NL);
                for (uint32_t lvl = 0; lvl < NL; lvl++) {
                    volatile tt_l1_ptr uint16_t* out_bf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        get_write_ptr(out_cb_index) + lvl * out_page_size);
                    uint32_t H = rt_H[lvl];
                    uint32_t W = rt_W[lvl];
                    float hs = (float)H * 0.5f;
                    float ws = (float)W * 0.5f;

                    for (uint32_t p = 0; p < NUM_PTS; p++) {
                        float h_img = (buf_gy[p] + 1.0f) * hs - 0.5f;
                        float w_img = (buf_gx[p] + 1.0f) * ws - 0.5f;
                        int32_t h0 = (int32_t)__builtin_floorf(h_img);
                        int32_t w0 = (int32_t)__builtin_floorf(w_img);
                        int32_t h1 = h0 + 1;
                        int32_t w1 = w0 + 1;
                        float dy = h_img - (float)h0;
                        float dx = w_img - (float)w0;

                        bool h0v = (h0 >= 0) && (h0 < (int32_t)H);
                        bool h1v = (h1 >= 0) && (h1 < (int32_t)H);
                        bool w0v = (w0 >= 0) && (w0 < (int32_t)W);
                        bool w1v = (w1 >= 0) && (w1 < (int32_t)W);
                        float wt_nw = (h0v && w0v) ? (1.0f - dy) * (1.0f - dx) : 0.0f;
                        float wt_ne = (h0v && w1v) ? (1.0f - dy) * dx : 0.0f;
                        float wt_sw = (h1v && w0v) ? dy * (1.0f - dx) : 0.0f;
                        float wt_se = (h1v && w1v) ? dy * dx : 0.0f;

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
            } else {
                // ===== Standard mode: output normalized [-1,1] grid coordinates =====
                // Batch CB: reserve all NC pages at once (first camera only)
                if (c == 0) {
                    cb_reserve_back(out_cb_index, NC);
                }
                uint32_t out_l1 = get_write_ptr(out_cb_index) + c * out_page_size;
                volatile tt_l1_ptr float* out = reinterpret_cast<volatile tt_l1_ptr float*>(out_l1);

                for (uint32_t p = 0; p < NUM_PTS; p++) {
                    uint32_t bf16_offset = p * (kp_page_size / sizeof(uint16_t));
                    float kx = bf16_to_f32(kp_bf16[bf16_offset + 0]);
                    float ky = bf16_to_f32(kp_bf16[bf16_offset + 1]);
                    float kz = bf16_to_f32(kp_bf16[bf16_offset + 2]);

                    float cos_kx = bf16_trunc(cos_yaw * kx);
                    float sin_ky = bf16_trunc(sin_yaw * ky);
                    float rx = bf16_trunc(cos_kx - sin_ky);
                    float sin_kx = bf16_trunc(sin_yaw * kx);
                    float cos_ky = bf16_trunc(cos_yaw * ky);
                    float ry = bf16_trunc(sin_kx + cos_ky);
                    float rz = kz;
                    float px = bf16_trunc(rx + cx);
                    float py = bf16_trunc(ry + cy);
                    float pz = bf16_trunc(rz + cz);

                    float proj_x = px * P0 + py * P1 + pz * P2  + P3;
                    float proj_y = px * P4 + py * P5 + pz * P6  + P7;
                    float proj_z = px * P8 + py * P9 + pz * P10 + P11;
                    float z_safe = proj_z > 1e-5f ? proj_z : 1e-5f;
                    float inv_z = 1.0f / z_safe;
                    float ndx = proj_x * inv_z;
                    float ndy = proj_y * inv_z;

                    float grid_x = ndx * inv_w * 2.0f - 1.0f;
                    float grid_y = ndy * inv_h * 2.0f - 1.0f;
                    if (grid_x < -2.0f) grid_x = -2.0f;
                    if (grid_x >  2.0f) grid_x =  2.0f;
                    if (grid_y < -2.0f) grid_y = -2.0f;
                    if (grid_y >  2.0f) grid_y =  2.0f;

                    out[p * 2 + 0] = grid_x;
                    out[p * 2 + 1] = grid_y;
                }
                // Batch CB: push all NC pages at once (last camera only)
                if (c == NC - 1) {
                    cb_push_back(out_cb_index, NC);
                }
            }
        }
    }
}
