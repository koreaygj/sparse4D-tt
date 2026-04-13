// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Reader kernel for TILE-based kps_project_fused (3-kernel architecture)
//
// Reads ROW_MAJOR inputs and constructs TILE-formatted data for compute kernel:
//   1. key_points [N, 13, 3] bf16 → TILE [32, 32] bf16 (one per anchor)
//   2. anchor [N, 1, 11] bf16 → rotation matrix TILE [32, 32] bf16
//   3. anchor → center broadcast TILE [32, 32] bf16
//   4. projection_mat [NC, 4, 4] f32 → P^T TILE [32, 32] f32 (per camera)
//   5. ones_col3 TILE [32, 32] f32 (constant)
//
// TILE face layout: 4 faces of 16×16
//   face0: rows 0-15, cols 0-15
//   face1: rows 0-15, cols 16-31
//   face2: rows 16-31, cols 0-15
//   face3: rows 16-31, cols 16-31
//   Element [r][c]: face[(r>=16)*2 + (c>=16)], offset (r%16)*16 + (c%16)
// =============================================================================

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"

// --- TILE element access helpers ---
// bf16 tile: 4 faces × 16×16 × 2 bytes = 2048 bytes
// f32 tile: 4 faces × 16×16 × 4 bytes = 4096 bytes

inline void tile_set_bf16(volatile tt_l1_ptr uint16_t* tile, uint32_t row, uint32_t col, uint16_t val) {
    uint32_t face = ((row >= 16) ? 2 : 0) + ((col >= 16) ? 1 : 0);
    uint32_t fr = row & 15;
    uint32_t fc = col & 15;
    tile[face * 256 + fr * 16 + fc] = val;
}

inline uint16_t tile_get_bf16(volatile tt_l1_ptr uint16_t* tile, uint32_t row, uint32_t col) {
    uint32_t face = ((row >= 16) ? 2 : 0) + ((col >= 16) ? 1 : 0);
    uint32_t fr = row & 15;
    uint32_t fc = col & 15;
    return tile[face * 256 + fr * 16 + fc];
}

inline void tile_set_f32(volatile tt_l1_ptr uint32_t* tile, uint32_t row, uint32_t col, float val) {
    uint32_t face = ((row >= 16) ? 2 : 0) + ((col >= 16) ? 1 : 0);
    uint32_t fr = row & 15;
    uint32_t fc = col & 15;
    uint32_t bits;
    __builtin_memcpy(&bits, &val, sizeof(float));
    tile[face * 256 + fr * 16 + fc] = bits;
}

inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = (bits & 0x10000u) ? 0x8000u : 0x7FFFu;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

// Zero a bf16 tile via NOC DMA loopback (2048 bytes)
// Requires a pre-zeroed 32-byte region at zero_src_addr
inline void zero_tile_bf16_dma(uint32_t tile_addr, uint32_t zero_src_addr) {
    // Write 32-byte chunks (64 writes for 2048 bytes)
    for (uint32_t off = 0; off < 2048; off += 32) {
        noc_async_write(zero_src_addr, get_noc_addr(tile_addr + off), 32);
    }
}

// Zero an f32 tile via NOC DMA loopback (4096 bytes)
inline void zero_tile_f32_dma(uint32_t tile_addr, uint32_t zero_src_addr) {
    for (uint32_t off = 0; off < 4096; off += 32) {
        noc_async_write(zero_src_addr, get_noc_addr(tile_addr + off), 32);
    }
}

// Volatile zero (fallback for first-time initialization)
inline void zero_region_volatile(uint32_t addr, uint32_t size_bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < size_bytes / 4; i++) p[i] = 0;
}

constexpr uint32_t ANC_X = 0, ANC_Y = 1, ANC_Z = 2;
constexpr uint32_t ANC_SIN_YAW = 6, ANC_COS_YAW = 7;

void kernel_main() {
    // Runtime args
    uint32_t kp_addr       = get_arg_val<uint32_t>(0);
    uint32_t anchor_addr   = get_arg_val<uint32_t>(1);
    uint32_t proj_addr     = get_arg_val<uint32_t>(2);
    uint32_t wh_addr       = get_arg_val<uint32_t>(3);
    uint32_t num_anchors   = get_arg_val<uint32_t>(4);
    uint32_t anchor_offset = get_arg_val<uint32_t>(5);

    // Compile-time args
    constexpr uint32_t cb_kp       = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rot      = get_compile_time_arg_val(1);
    constexpr uint32_t cb_center   = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ones     = get_compile_time_arg_val(3);
    constexpr uint32_t cb_proj     = get_compile_time_arg_val(4);
    constexpr uint32_t NC          = get_compile_time_arg_val(5);
    constexpr uint32_t NUM_PTS     = get_compile_time_arg_val(6);
    constexpr uint32_t kp_page_size    = get_compile_time_arg_val(7);
    constexpr uint32_t anchor_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t proj_page_size  = get_compile_time_arg_val(9);
    constexpr uint32_t wh_page_size    = get_compile_time_arg_val(10);
    constexpr uint32_t bf16_tile_size  = get_compile_time_arg_val(11);  // 2048
    constexpr uint32_t f32_tile_size   = get_compile_time_arg_val(12);  // 4096

    constexpr auto kp_args = TensorAccessorArgs<13>();
    constexpr auto anchor_args = TensorAccessorArgs<kp_args.next_compile_time_args_offset()>();
    constexpr auto proj_args = TensorAccessorArgs<anchor_args.next_compile_time_args_offset()>();
    constexpr auto wh_args = TensorAccessorArgs<proj_args.next_compile_time_args_offset()>();

    const auto kp_accessor = TensorAccessor(kp_args, kp_addr, kp_page_size);
    const auto anchor_accessor = TensorAccessor(anchor_args, anchor_addr, anchor_page_size);
    const auto proj_accessor = TensorAccessor(proj_args, proj_addr, proj_page_size);
    const auto wh_accessor = TensorAccessor(wh_args, wh_addr, wh_page_size);

    // Page stride for proj (alignment may pad beyond 4 floats)
    constexpr uint32_t proj_page_floats = proj_page_size / sizeof(float);

    // Scratch CB for raw DRAM page reads (anchor, kp, proj)
    constexpr uint32_t cb_scratch = tt::CBIndex::c_18;
    uint32_t scratch_base = get_write_ptr(cb_scratch);

    // Pre-zero a 32-byte region in scratch for NOC DMA zero fills
    uint32_t zero_src = scratch_base;
    zero_region_volatile(zero_src, 32);
    uint32_t data_scratch = scratch_base + 32;

    // Build ones_col3 tile (constant, pushed once)
    // ones_col3[r][3] = 1.0f for all r, rest = 0
    cb_reserve_back(cb_ones, 1);
    uint32_t ones_addr = get_write_ptr(cb_ones);
    volatile tt_l1_ptr uint32_t* ones_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ones_addr);
    zero_tile_f32_dma(ones_addr, zero_src);
    noc_async_write_barrier();
    // 1.0f = 0x3F800000
    for (uint32_t r = 0; r < 32; r++) {
        tile_set_f32(ones_tile, r, 3, 1.0f);
    }
    cb_push_back(cb_ones, 1);

    // Pre-load projection matrices and build P^T tiles for each camera
    // P^T[k][j] = P[j][k] — we transpose so matmul(pts_homo, P^T) gives pts @ P^T

    bool slots_initialized = false;

    for (uint32_t a_idx = 0; a_idx < num_anchors; a_idx++) {
        uint32_t a = anchor_offset + a_idx;

        // --- Batch read: anchor + key_points in one NOC burst ---
        uint32_t anc_scratch = data_scratch;
        uint32_t kp_scratch = data_scratch + anchor_page_size;

        // Issue all DRAM reads at once (anchor + 13 kp pages)
        noc_async_read(anchor_accessor.get_noc_addr(a), anc_scratch, anchor_page_size);
        uint32_t kp_start_page = a * NUM_PTS;
        for (uint32_t p = 0; p < NUM_PTS; p++) {
            noc_async_read(kp_accessor.get_noc_addr(kp_start_page + p), kp_scratch + p * kp_page_size, kp_page_size);
        }

        // While reads are in flight, prepare CB tiles (zero on first use)
        cb_reserve_back(cb_rot, 1);
        uint32_t rot_addr = get_write_ptr(cb_rot);
        volatile tt_l1_ptr uint16_t* rot_tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rot_addr);
        if (!slots_initialized) {
            zero_tile_bf16_dma(rot_addr, zero_src);
        }

        cb_reserve_back(cb_center, 1);
        uint32_t center_addr = get_write_ptr(cb_center);
        volatile tt_l1_ptr uint16_t* center_tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(center_addr);
        if (!slots_initialized) {
            zero_tile_bf16_dma(center_addr, zero_src);
        }

        cb_reserve_back(cb_kp, 1);
        uint32_t kp_addr = get_write_ptr(cb_kp);
        volatile tt_l1_ptr uint16_t* kp_tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(kp_addr);
        if (!slots_initialized) {
            zero_tile_bf16_dma(kp_addr, zero_src);
        }

        // Wait for all reads + zero DMAs
        noc_async_read_barrier();
        if (!slots_initialized) noc_async_write_barrier();

        // Now process anchor data
        volatile tt_l1_ptr uint16_t* anc_bf16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(anc_scratch);
        uint16_t cos_yaw_bf16 = anc_bf16[ANC_COS_YAW];
        uint16_t sin_yaw_bf16 = anc_bf16[ANC_SIN_YAW];
        uint16_t cx_bf16 = anc_bf16[ANC_X];
        uint16_t cy_bf16 = anc_bf16[ANC_Y];
        uint16_t cz_bf16 = anc_bf16[ANC_Z];
        float neg_sin = -bf16_to_f32(sin_yaw_bf16);
        uint16_t neg_sin_bf16 = f32_to_bf16(neg_sin);

        // Build rotation matrix tile (5 scatter writes)
        tile_set_bf16(rot_tile, 0, 0, cos_yaw_bf16);
        tile_set_bf16(rot_tile, 0, 1, sin_yaw_bf16);
        tile_set_bf16(rot_tile, 1, 0, neg_sin_bf16);
        tile_set_bf16(rot_tile, 1, 1, cos_yaw_bf16);
        tile_set_bf16(rot_tile, 2, 2, 0x3F80);
        cb_push_back(cb_rot, 1);

        // Build center broadcast tile
        for (uint32_t r = 0; r < NUM_PTS; r++) {
            tile_set_bf16(center_tile, r, 0, cx_bf16);
            tile_set_bf16(center_tile, r, 1, cy_bf16);
            tile_set_bf16(center_tile, r, 2, cz_bf16);
        }
        cb_push_back(cb_center, 1);

        volatile tt_l1_ptr uint16_t* kp_raw = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(kp_scratch);
        uint32_t kp_page_elems = kp_page_size / sizeof(uint16_t);
        for (uint32_t p = 0; p < NUM_PTS; p++) {
            uint16_t kx = kp_raw[p * kp_page_elems + 0];
            uint16_t ky = kp_raw[p * kp_page_elems + 1];
            uint16_t kz = kp_raw[p * kp_page_elems + 2];
            tile_set_bf16(kp_tile, p, 0, kx);
            tile_set_bf16(kp_tile, p, 1, ky);
            tile_set_bf16(kp_tile, p, 2, kz);
        }
        cb_push_back(cb_kp, 1);

        // --- Build P^T tiles for each camera (read on-the-fly) ---
        // P is [NC, 4, 4] ROW_MAJOR f32, each row = 4 floats = 1 page
        for (uint32_t c = 0; c < NC; c++) {
            cb_reserve_back(cb_proj, 1);
            uint32_t proj_tile_addr = get_write_ptr(cb_proj);
            volatile tt_l1_ptr uint32_t* proj_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(proj_tile_addr);
            if (!slots_initialized) {
                zero_tile_f32_dma(proj_tile_addr, zero_src);
            }

            // Read 4 rows of P[c] into scratch area (overlap with zero DMA)
            uint32_t proj_scratch = data_scratch + anchor_page_size + NUM_PTS * kp_page_size;
            for (uint32_t row = 0; row < 4; row++) {
                uint64_t noc_addr = proj_accessor.get_noc_addr(c * 4 + row);
                noc_async_read(noc_addr, proj_scratch + row * proj_page_size, proj_page_size);
            }
            noc_async_read_barrier();
            if (!slots_initialized) noc_async_write_barrier();

            volatile tt_l1_ptr float* pdata = reinterpret_cast<volatile tt_l1_ptr float*>(proj_scratch);
            constexpr uint32_t ppf = proj_page_size / sizeof(float);
            for (uint32_t row = 0; row < 4; row++) {
                for (uint32_t col = 0; col < 4; col++) {
                    float val = pdata[row * ppf + col];
                    tile_set_f32(proj_tile, col, row, val);
                }
            }
            cb_push_back(cb_proj, 1);
        }

        // After first full cycle, CB slots are zeroed — only need scatter writes
        if (!slots_initialized && a_idx >= 1) {
            slots_initialized = true;
        }
    }
}
