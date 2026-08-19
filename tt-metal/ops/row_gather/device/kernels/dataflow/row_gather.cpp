// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Paired row gather: out_a[r,:] = src_a[idx[r],:], out_b[r,:] = src_b[idx[r],:].
//
// Replaces ttnn.gather for the instance bank's top-k selection, which ran on
// 4 cores at 1.1 GB/s — pure NOC round-trip latency, 655 us/call — plus a
// typecast/reshape/to_layout/repeat_interleave chain per call just to expand
// the indices into gather's format. This kernel takes topk_select's uint32
// ROW_MAJOR indices directly and copies rows in parallel across cores with
// deep async pipelining; feature and anchor ride the same index read.
//
// TILE row addressing: a 32x32 tile stores faces f0(r0-15,c0-15) f1(r0-15,
// c16-31) f2(r16-31,c0-15) f3(r16-31,c16-31), each row-major 16 wide. Logical
// row ls of one tile is two contiguous segments of 16 elements:
//   ls < 16 : f0 + ls*16        and  f1 + ls*16   (f1 base = 256 elems)
//   ls >= 16: f2 + (ls-16)*16   and  f3 + (ls-16)*16 (bases 512, 768)

#include <stdint.h>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

// Reads and writes are issued for a WHOLE row before any barrier: one
// read-barrier per row instead of one per tile (39 col-tiles for the gws
// weight permute made per-tile barriers latency-bound).
template <uint32_t E>
FORCE_INLINE void read_row_seg(const uint64_t tile_base, const uint32_t ls, uint32_t l1_buf) {
    const uint32_t half = (ls < 16) ? 0u : 512u * E;
    const uint32_t off = ((ls & 15u) * 16u) * E;
    noc_async_read(tile_base + half + off, l1_buf, 16 * E);
    noc_async_read(tile_base + half + 256u * E + off, l1_buf + 16 * E, 16 * E);
}

template <uint32_t E>
FORCE_INLINE void write_row_seg(uint32_t l1_buf, const uint64_t tile_base, const uint32_t lr) {
    const uint32_t half = (lr < 16) ? 0u : 512u * E;
    const uint32_t off = ((lr & 15u) * 16u) * E;
    noc_async_write(l1_buf, tile_base + half + off, 16 * E);
    noc_async_write(l1_buf + 16 * E, tile_base + half + 256u * E + off, 16 * E);
}

void kernel_main() {
    constexpr uint32_t K     = get_compile_time_arg_val(0);   // rows to gather
    constexpr uint32_t CT_A  = get_compile_time_arg_val(1);   // col tiles of a (8)
    constexpr uint32_t CT_B  = get_compile_time_arg_val(2);   // col tiles of b (1)
    constexpr uint32_t EA    = get_compile_time_arg_val(3);   // elem bytes a (2)
    constexpr uint32_t EB    = get_compile_time_arg_val(4);   // elem bytes b (4)
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(5);
    constexpr uint32_t CB_ROW = get_compile_time_arg_val(6);
    // ROW_MAJOR pair-a: rows are whole sticks — one read + one write per row
    // instead of per-tile face segments (used by the gws output un-permute,
    // whose non-tile-aligned slice lands in RM).
    constexpr uint32_t A_RM       = get_compile_time_arg_val(7);
    constexpr uint32_t A_STICK_SZ = get_compile_time_arg_val(8);

    constexpr auto sa_args = TensorAccessorArgs<9>();
    constexpr auto sb_args = TensorAccessorArgs<sa_args.next_compile_time_args_offset()>();
    constexpr auto ix_args = TensorAccessorArgs<sb_args.next_compile_time_args_offset()>();
    constexpr auto oa_args = TensorAccessorArgs<ix_args.next_compile_time_args_offset()>();
    constexpr auto ob_args = TensorAccessorArgs<oa_args.next_compile_time_args_offset()>();

    const uint32_t sa_addr = get_arg_val<uint32_t>(0);
    const uint32_t sb_addr = get_arg_val<uint32_t>(1);
    const uint32_t ix_addr = get_arg_val<uint32_t>(2);
    const uint32_t oa_addr = get_arg_val<uint32_t>(3);
    const uint32_t ob_addr = get_arg_val<uint32_t>(4);
    const uint32_t row0    = get_arg_val<uint32_t>(5);
    const uint32_t nrows   = get_arg_val<uint32_t>(6);

    const uint32_t ta_bytes = A_RM ? A_STICK_SZ : 32 * 32 * EA;
    const uint32_t tb_bytes = 32 * 32 * EB;
    const auto sa = TensorAccessor(sa_args, sa_addr, ta_bytes);
    const auto oa = TensorAccessor(oa_args, oa_addr, ta_bytes);
    const auto sb = TensorAccessor(sb_args, sb_addr, tb_bytes);
    const auto ob = TensorAccessor(ob_args, ob_addr, tb_bytes);
    const auto ix = TensorAccessor(ix_args, ix_addr, K * 4);

    // whole index stick once (<= 2.4 KB)
    cb_reserve_back(CB_IDX, 1);
    const uint32_t ixb = get_write_ptr(CB_IDX);
    noc_async_read(ix.get_noc_addr(0), ixb, K * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx = (volatile tt_l1_ptr uint32_t*)ixb;

    cb_reserve_back(CB_ROW, 1);
    const uint32_t rb = get_write_ptr(CB_ROW);

    for (uint32_t i = 0; i < nrows; i++) {
        const uint32_t r = row0 + i;      // destination row
        const uint32_t s = idx[r];        // source row
        const uint32_t st = s >> 5, ls = s & 31u;
        const uint32_t dt = r >> 5, lr = r & 31u;
        uint32_t buf = rb;
        if constexpr (A_RM) {
            noc_async_read(sa.get_noc_addr(s), buf, A_STICK_SZ);
            buf += A_STICK_SZ;
        } else {
            for (uint32_t c = 0; c < CT_A; c++) {
                read_row_seg<EA>(sa.get_noc_addr(st * CT_A + c), ls, buf);
                buf += 32 * EA;
            }
        }
        for (uint32_t c = 0; c < CT_B; c++) {
            read_row_seg<EB>(sb.get_noc_addr(st * CT_B + c), ls, buf);
            buf += 32 * EB;
        }
        noc_async_read_barrier();
        buf = rb;
        if constexpr (A_RM) {
            noc_async_write(buf, oa.get_noc_addr(r), A_STICK_SZ);
            buf += A_STICK_SZ;
        } else {
            for (uint32_t c = 0; c < CT_A; c++) {
                write_row_seg<EA>(buf, oa.get_noc_addr(dt * CT_A + c), lr);
                buf += 32 * EA;
            }
        }
        for (uint32_t c = 0; c < CT_B; c++) {
            write_row_seg<EB>(buf, ob.get_noc_addr(dt * CT_B + c), lr);
            buf += 32 * EB;
        }
        // the row buffer is reused next iteration; wait until the NIU has
        // pulled every write payload out of L1 (cheaper than a full ack)
        noc_async_writes_flushed();
    }
    noc_async_write_barrier();
}
