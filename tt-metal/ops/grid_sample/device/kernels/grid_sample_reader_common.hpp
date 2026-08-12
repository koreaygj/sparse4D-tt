// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/numeric/bfloat16.h"

#define ALWI inline __attribute__((always_inline))

// Constants shared between interleaved and sharded kernels
constexpr uint32_t PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6;          // For bilinear mode
constexpr uint32_t PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST = 2;  // For nearest mode
constexpr uint32_t STANDARD_GRID_ELEMENTS_PER_POINT = 2;

// Data type constants (from ttnn/api/ttnn/tensor/types.hpp DataType enum)
constexpr uint32_t DTYPE_BFLOAT16 = 0;
constexpr uint32_t DTYPE_FLOAT32 = 1;
// UINT16 is a CONTAINER for a Q14 fixed-point coordinate, not an unsigned value. The
// coordinate is clamped to [-2, 2], so a float's exponent bits buy nothing while the
// bilinear weight it feeds is the coordinate's fractional part and wants uniform
// absolute precision. Measured relative error in the sampled value: bf16 3.8e-2,
// Q14 8.6e-4, against 1.0e-3 from the feature map's own bf16 storage.
// The shift is shared with kps_project_fused (which writes it) and grid_compact.
constexpr uint32_t DTYPE_UINT16 = 6;
constexpr int32_t GRID_FIXED_SHIFT = 14;
constexpr float GRID_FIXED_INV = 1.0f / (float)(1 << GRID_FIXED_SHIFT);

// Utility functions
ALWI bool is_coordinate_valid(int32_t coord, uint32_t max_size) {
    return (coord >= 0) && (coord < static_cast<int32_t>(max_size));
}

ALWI void fill_four_val(uint32_t begin_addr, uint16_t val, uint16_t val1, uint16_t val2, uint16_t val3) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    ptr[0] = (val | (val1 << 16));
    ptr[1] = (val2 | (val3 << 16));
}

// Grid coordinate reading functions
template <uint32_t grid_dtype, bool use_precomputed_grid>
struct GridCoordinateReader {
    // Read grid coordinates and return corner coordinates and weights
    template <typename GridPtrType>
    ALWI static void read_grid_point(
        GridPtrType grid_ptr,
        uint32_t grid_idx,
        uint32_t grid_k,
        float height_scale,
        float height_offset,
        float width_scale,
        float width_offset,
        uint32_t input_height,
        uint32_t input_width,
        int32_t& h0,
        int32_t& h1,
        int32_t& w0,
        int32_t& w1,
        uint16_t& weight_nw_bf,
        uint16_t& weight_ne_bf,
        uint16_t& weight_sw_bf,
        uint16_t& weight_se_bf) {
        if constexpr (use_precomputed_grid) {
            // FIELD-MAJOR within the stick: [h0 x K][w0 x K][nw x K][ne x K][sw x K][se x K],
            // K being the grid batching factor. Grouping by field rather than by point lets
            // the producer (grid_precompute's writer) emit each field as one contiguous run
            // instead of interleaving two source tiles two bytes at a time — the interleaved
            // form cost that writer ~40 instructions per value. This reader does the same
            // six loads either way; only the offsets differ.
            //
            // NOTE this diverges from upstream's interleaved layout, and from what the host
            // prepare_grid_sample_grid emits. In this repo the only producer of precomputed
            // grids is ttnn.grid_precompute, which emits this layout.
            const int16_t h0_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[grid_idx]);
            const int16_t w0_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[grid_k + grid_idx]);

            h0 = static_cast<int32_t>(h0_raw);
            w0 = static_cast<int32_t>(w0_raw);
            h1 = h0 + 1;
            w1 = w0 + 1;

            weight_nw_bf = grid_ptr[2 * grid_k + grid_idx];
            weight_ne_bf = grid_ptr[3 * grid_k + grid_idx];
            weight_sw_bf = grid_ptr[4 * grid_k + grid_idx];
            weight_se_bf = grid_ptr[5 * grid_k + grid_idx];
        } else {
            // Each regular grid entry has 2 values: x, y coordinates
            float h_coord_rel, w_coord_rel;
            if constexpr (grid_dtype == DTYPE_FLOAT32) {
                // For FLOAT32 grid, each coordinate is a 32-bit float
                volatile tt_l1_ptr float* float_data = reinterpret_cast<volatile tt_l1_ptr float*>(grid_ptr);
                const uint32_t float_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
                w_coord_rel = float_data[float_offset + 0];  // x coordinate
                h_coord_rel = float_data[float_offset + 1];  // y coordinate
            } else if constexpr (grid_dtype == DTYPE_UINT16) {
                // Q14 fixed point: reinterpret the 16 bits as signed and scale.
                const uint32_t coordinate_pair_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
                const uint16_t h_raw = grid_ptr[coordinate_pair_offset + 1];  // y coordinate
                const uint16_t w_raw = grid_ptr[coordinate_pair_offset + 0];  // x coordinate
                h_coord_rel = static_cast<float>(static_cast<int16_t>(h_raw)) * GRID_FIXED_INV;
                w_coord_rel = static_cast<float>(static_cast<int16_t>(w_raw)) * GRID_FIXED_INV;
            } else {
                // For BFLOAT16 grid, read as uint16 and convert
                const uint32_t coordinate_pair_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
                const uint16_t h_coord_raw = grid_ptr[coordinate_pair_offset + 1];  // y coordinate
                const uint16_t w_coord_raw = grid_ptr[coordinate_pair_offset + 0];  // x coordinate
                h_coord_rel = bf16_to_fp32(h_coord_raw);
                w_coord_rel = bf16_to_fp32(w_coord_raw);
            }

            const float h_coord_image = h_coord_rel * height_scale + height_offset;
            const float w_coord_image = w_coord_rel * width_scale + width_offset;

            h0 = static_cast<int32_t>(floor(h_coord_image));
            h1 = h0 + 1;
            w0 = static_cast<int32_t>(floor(w_coord_image));
            w1 = w0 + 1;

            // Calculate bilinear interpolation weights
            const float h0_f = static_cast<float>(h0);
            const float w0_f = static_cast<float>(w0);

            const float h_frac = h_coord_image - h0_f;
            const float w_frac = w_coord_image - w0_f;
            const float h_frac_inv = 1.0f - h_frac;
            const float w_frac_inv = 1.0f - w_frac;

            // Boundary checks
            const bool h0_valid = is_coordinate_valid(h0, input_height);
            const bool h1_valid = is_coordinate_valid(h1, input_height);
            const bool w0_valid = is_coordinate_valid(w0, input_width);
            const bool w1_valid = is_coordinate_valid(w1, input_width);

            const float weight_nw = (h0_valid && w0_valid) ? (h_frac_inv * w_frac_inv) : 0.0f;  // North-West
            const float weight_ne = (h0_valid && w1_valid) ? (h_frac_inv * w_frac) : 0.0f;      // North-East
            const float weight_sw = (h1_valid && w0_valid) ? (h_frac * w_frac_inv) : 0.0f;      // South-West
            const float weight_se = (h1_valid && w1_valid) ? (h_frac * w_frac) : 0.0f;          // South-East

            weight_nw_bf = fp32_to_bf16_truncate(weight_nw);
            weight_ne_bf = fp32_to_bf16_truncate(weight_ne);
            weight_sw_bf = fp32_to_bf16_truncate(weight_sw);
            weight_se_bf = fp32_to_bf16_truncate(weight_se);
        }
    }
};

// Input data reading template - handles both tensor accessor and direct NOC reads.
// Reads a single chunk slice of the 4 corner sticks for one grid point.
// - read_bytes: bytes to copy per corner (chunk width)
// - write_stride: distance between consecutive corners in the destination CB page
// - src_byte_offset: byte offset within the source stick (= c_i * input_chunk_nbytes)
template <typename TensorAccessorT>
ALWI void read_four_corner_inputs(
    Noc noc,
    const TensorAccessorT& input_tensor_accessor,
    uint32_t batch_offset,
    uint32_t input_width,
    uint32_t read_bytes,
    uint32_t write_stride,
    uint32_t src_byte_offset,
    int32_t h0,
    int32_t h1,
    int32_t w0,
    int32_t w1,
    uint32_t input_height,
    experimental::CB input_cb) {
    // Boundary checks (recompute for performance)
    const bool h0_valid = is_coordinate_valid(h0, input_height);
    const bool h1_valid = is_coordinate_valid(h1, input_height);
    const bool w0_valid = is_coordinate_valid(w0, input_width);
    const bool w1_valid = is_coordinate_valid(w1, input_width);

    uint32_t write_offset = 0;

    // Read 4 corner input sticks via NOC from remote input tensor
    if (h0_valid && w0_valid) {
        const uint32_t north_west_stick_index = batch_offset + (h0 * input_width) + w0;
        noc.async_read(
            input_tensor_accessor,
            input_cb,
            read_bytes,
            {.page_id = north_west_stick_index, .offset_bytes = src_byte_offset},
            {.offset_bytes = write_offset});
    }
    write_offset += write_stride;

    if (h0_valid && w1_valid) {
        const uint32_t north_east_stick_index = batch_offset + (h0 * input_width) + w1;
        noc.async_read(
            input_tensor_accessor,
            input_cb,
            read_bytes,
            {.page_id = north_east_stick_index, .offset_bytes = src_byte_offset},
            {.offset_bytes = write_offset});
    }
    write_offset += write_stride;

    if (h1_valid && w0_valid) {
        const uint32_t south_west_stick_index = batch_offset + (h1 * input_width) + w0;
        noc.async_read(
            input_tensor_accessor,
            input_cb,
            read_bytes,
            {.page_id = south_west_stick_index, .offset_bytes = src_byte_offset},
            {.offset_bytes = write_offset});
    }
    write_offset += write_stride;

    if (h1_valid && w1_valid) {
        const uint32_t south_east_stick_index = batch_offset + (h1 * input_width) + w1;
        noc.async_read(
            input_tensor_accessor,
            input_cb,
            read_bytes,
            {.page_id = south_east_stick_index, .offset_bytes = src_byte_offset},
            {.offset_bytes = write_offset});
    }
}

template <typename TensorAccessorT>
ALWI void read_four_corner_inputs_with_fill(
    Noc noc,
    const TensorAccessorT& input_tensor_accessor,
    uint32_t batch_offset,
    uint32_t input_width,
    uint32_t input_stick_nbytes,
    int32_t h0,
    int32_t h1,
    int32_t w0,
    int32_t w1,
    uint32_t input_height,
    experimental::CB input_cb,
    uint32_t fill_stick_addr) {
    const bool h0_valid = is_coordinate_valid(h0, input_height);
    const bool h1_valid = is_coordinate_valid(h1, input_height);
    const bool w0_valid = is_coordinate_valid(w0, input_width);
    const bool w1_valid = is_coordinate_valid(w1, input_width);

    UnicastEndpoint self_ep;
    const auto fill_src = experimental::local_addr(fill_stick_addr, noc.get_noc_id());
    uint32_t write_offset = 0;

    if (h0_valid && w0_valid) {
        const uint32_t north_west_stick_index = batch_offset + (h0 * input_width) + w0;
        noc.async_read(
            input_tensor_accessor,
            input_cb,
            input_stick_nbytes,
            {.page_id = north_west_stick_index},
            {.offset_bytes = write_offset});
    } else {
        noc.async_read(self_ep, input_cb, input_stick_nbytes, fill_src, {.offset_bytes = write_offset});
    }
    write_offset += input_stick_nbytes;

    if (h0_valid && w1_valid) {
        const uint32_t north_east_stick_index = batch_offset + (h0 * input_width) + w1;
        noc.async_read(
            input_tensor_accessor,
            input_cb,
            input_stick_nbytes,
            {.page_id = north_east_stick_index},
            {.offset_bytes = write_offset});
    } else {
        noc.async_read(self_ep, input_cb, input_stick_nbytes, fill_src, {.offset_bytes = write_offset});
    }
    write_offset += input_stick_nbytes;

    if (h1_valid && w0_valid) {
        const uint32_t south_west_stick_index = batch_offset + (h1 * input_width) + w0;
        noc.async_read(
            input_tensor_accessor,
            input_cb,
            input_stick_nbytes,
            {.page_id = south_west_stick_index},
            {.offset_bytes = write_offset});
    } else {
        noc.async_read(self_ep, input_cb, input_stick_nbytes, fill_src, {.offset_bytes = write_offset});
    }
    write_offset += input_stick_nbytes;

    if (h1_valid && w1_valid) {
        const uint32_t south_east_stick_index = batch_offset + (h1 * input_width) + w1;
        noc.async_read(
            input_tensor_accessor,
            input_cb,
            input_stick_nbytes,
            {.page_id = south_east_stick_index},
            {.offset_bytes = write_offset});
    } else {
        noc.async_read(self_ep, input_cb, input_stick_nbytes, fill_src, {.offset_bytes = write_offset});
    }
}

// Process single grid point - common logic for both interleaved and sharded.
//
// When in_nblocks_c > 1 the channel dimension is split into chunks of at most
// input_chunk_nbytes bytes (= MAX_TILES_PER_REDUCTION * TILE_WIDTH * elem_size). For each chunk we
// reserve one input CB page, NOC-read the c_i-th slice of all 4 corner sticks, and push that page
// to the compute kernel, which iterates over the same in_nblocks_c chunks. The scalar (bilinear
// weights) CB is pushed once per grid point since the weights are shared across chunks.
template <
    uint32_t grid_dtype,
    bool use_precomputed_grid,
    bool align_corners,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t input_stick_nbytes,
    uint32_t in_nblocks_c,
    uint32_t input_chunk_nbytes,
    bool last_chunk_partial,
    uint32_t input_cb_index,
    uint32_t scalar_cb_index,
    typename TensorAccessor,
    typename GridPtrType>
ALWI void process_grid_point(
    Noc noc,
    experimental::CB input_cb,
    experimental::CB scalar_cb,
    GridPtrType grid_ptr,
    uint32_t grid_idx,
    uint32_t grid_k,
    const TensorAccessor& input_tensor_accessor,
    uint32_t batch_offset) {
    // PyTorch grid_sample coordinate convention:
    //   align_corners=True : maps grid in [-1, 1] to image positions [0, size - 1]
    //                        => coord_image = ((grid + 1) / 2) * (size - 1)
    //                                       = grid * (size - 1) / 2 + (size - 1) / 2
    //   align_corners=False: maps grid in [-1, 1] to image positions [-0.5, size - 0.5]
    //                        => coord_image = ((grid + 1) / 2) * size - 0.5
    //                                       = grid * size / 2 + size / 2 - 0.5
    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);
    constexpr float height_scale =
        align_corners ? (input_height_f > 1.0f ? (input_height_f - 1.0f) * 0.5f : 0.0f) : input_height_f * 0.5f;
    constexpr float height_offset = align_corners ? (input_height_f > 1.0f ? (input_height_f - 1.0f) * 0.5f : 0.0f)
                                                  : (input_height_f * 0.5f - 0.5f);
    constexpr float width_scale =
        align_corners ? (input_width_f > 1.0f ? (input_width_f - 1.0f) * 0.5f : 0.0f) : input_width_f * 0.5f;
    constexpr float width_offset =
        align_corners ? (input_width_f > 1.0f ? (input_width_f - 1.0f) * 0.5f : 0.0f) : (input_width_f * 0.5f - 0.5f);

    int32_t h0, h1, w0, w1;
    uint16_t weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf;

    // Read grid coordinates and compute weights
    GridCoordinateReader<grid_dtype, use_precomputed_grid>::template read_grid_point(
        grid_ptr,
        grid_idx,
        grid_k,
        height_scale,
        height_offset,
        width_scale,
        width_offset,
        input_height,
        input_width,
        h0,
        h1,
        w0,
        w1,
        weight_nw_bf,
        weight_ne_bf,
        weight_sw_bf,
        weight_se_bf);

    // For precomputed grid, we need to compute boundary checks here
    // since they weren't computed in the coordinate reading phase
    if constexpr (use_precomputed_grid) {
        const bool h0_valid = is_coordinate_valid(h0, input_height);
        const bool h1_valid = is_coordinate_valid(h1, input_height);
        const bool w0_valid = is_coordinate_valid(w0, input_width);
        const bool w1_valid = is_coordinate_valid(w1, input_width);

        // Zero out weights for invalid coordinates
        if (!(h0_valid && w0_valid)) {
            weight_nw_bf = 0;
        }
        if (!(h0_valid && w1_valid)) {
            weight_ne_bf = 0;
        }
        if (!(h1_valid && w0_valid)) {
            weight_sw_bf = 0;
        }
        if (!(h1_valid && w1_valid)) {
            weight_se_bf = 0;
        }
    }

    // Store bilinear interpolation weights for this grid point. The same weights apply to every
    // channel chunk, so we push the scalar CB only once.
    scalar_cb.reserve_back(1);
    const uint32_t l1_write_scalar_addr = scalar_cb.get_write_ptr();
    fill_four_val(l1_write_scalar_addr, weight_nw_bf, weight_ne_bf, weight_sw_bf, weight_se_bf);
    scalar_cb.push_back(1);

    // Iterate over channel chunks. For the common case in_nblocks_c == 1 the loop runs once and the
    // behavior matches the original non-chunked reader (chunk_bytes == input_stick_nbytes,
    // src_byte_offset == 0, write_stride == input_stick_nbytes). last_chunk_partial is computed
    // host-side to mirror compute_pool_2d's tile-reduce condition so the per-stick write stride
    // matches the unpacker's tiles_to_reduce; computing it independently here would silently diverge
    // if the host ever lifted the padded_C % TILE_WIDTH == 0 invariant.
    constexpr uint32_t last_chunk_idx = in_nblocks_c - 1;
    constexpr uint32_t partial_chunk_nbytes = input_stick_nbytes - last_chunk_idx * input_chunk_nbytes;
    constexpr uint32_t base_write_stride = (in_nblocks_c > 1) ? input_chunk_nbytes : input_stick_nbytes;

    for (uint32_t c_i = 0; c_i < in_nblocks_c; ++c_i) {
        const uint32_t src_byte_offset = c_i * input_chunk_nbytes;
        const uint32_t chunk_bytes = (c_i == last_chunk_idx) ? partial_chunk_nbytes : input_chunk_nbytes;
        const uint32_t write_stride = (last_chunk_partial && c_i == last_chunk_idx) ? chunk_bytes : base_write_stride;

        input_cb.reserve_back(1);

        read_four_corner_inputs(
            noc,
            input_tensor_accessor,
            batch_offset,
            input_width,
            chunk_bytes,
            write_stride,
            src_byte_offset,
            h0,
            h1,
            w0,
            w1,
            input_height,
            input_cb);

        noc.async_read_barrier();
        input_cb.push_back(1);
    }
}
