# Update log

Per-change measurements. Figures are from this machine unless stated otherwise; "no change"
means measured and found within noise, not unmeasured.

Headline results live in the [README](../README.md); this file records how they were reached.

---

## v3

Full val, 6019 samples, `sparse4dv3_r50.pth`: mAP 0.4019 -> **0.4515**, NDS 0.5186 ->
**0.5553**. Latency 90.7 -> **65.2 ms/sample** (11.02 -> 15.34 FPS).

### Accuracy

**`42a847b` — all-reduce the softmax denominator across the mesh**

- SPMD gives each device 3 of 6 cameras, so its sampling-point axis is 156 of 312, and
  `_softmax_clp` built the denominator from its own 156
- Each device's weights summed to 1 alone (row sums 1.0063 each, 2.01 together), so the
  post-fusion `all_reduce` added two complete distributions
- Fix reduces the denominator across devices **and** the max shift with it — `exp(l - m0)`
  and `exp(l - m1)` are not summable, so both devices must subtract the same constant.
  `ttnn.all_reduce` is Sum-only, so it uses the mean of the per-device maxima; any common
  constant is exact
- **mAP 0.4019 -> 0.4476**, DFA output PCC 0.9883 -> 0.9999, no measurable speed cost
- Verified identical on the pure-TT-NN fallback path (`TT_WT_COMPACT=0`)

**`8ba12e4` — keep the projection matrix in fp32**

- The cache did `pad -> typecast(float32) -> slice` with the typecast **last**, so pages
  said fp32 while the values had already been rounded to bf16
- Matrix entries reach 667, where bf16 leaves 1.3 of absolute error, and the kernel divides
  by depth — the largest single term in the grid's disagreement with PyTorch
- Sampling grid error **0.179 -> 0.035 px** (level 0)
- **mAP unchanged** (0.4023 -> 0.4019, noise). The sampling path was already PCC 0.99998, so
  the correctness matters but the accuracy did not move
- Only the camera encoder keeps a bf16 view — a matmul chain that never reaches a pixel

**`33778e7` — widen the anchor accumulator to fp32**

- The anchor is a residual stream (`anchor + delta` at every layer), so it accumulates
  absolute error; bf16 gives constant *relative* error
- mAP 0.4008 -> 0.4023
- Weight upload moved to host-side bf16 in the same commit: bit-identical, -0.15 ms/frame,
  -53 ms startup

**`ca4f025` — scale the compaction budget with the camera count**

**`d5b833a` — grid_sample padding clamp used the input batch, not the grid's**

### Speed

**fold the temporal anchor projection into one affine matmul** (branch perf/topk-kernel)

- The projection is linear in every anchor field, so the 13-op slice/matmul/concat
  chain (plus 4 h2d uploads/frame) folds into anchor @ A + b with A built on host
  from the frame's ego pose. Wall 67.2 -> 66.4 ms
- Full val: mAP 0.4482 -> **0.4515**, NDS **0.5553**, mATE 0.5615 -> 0.5492, mAOE
  0.4851 -> 0.4710 — the fold is MORE precise than the chain it replaced (one
  fp32 matmul instead of repeated bf16 intermediate roundings), so the temporal
  anchors improve and accuracy sets a new project best

**make linear's relu a real fused epilogue** (branch perf/topk-kernel)

- `ttnn.linear(activation="relu")` silently does NOT fuse without a user core
  grid: matmul.cpp runs the activation as a separate unary op afterwards. The
  encoder and refinement chains believed they were fused — the profile showed
  168 stray RELU ops/frame (1.75 ms device + ~0.7 ms host dispatch)
- Fix is `core_grid=ttnn.CoreGrid(y=8, x=8)` on those linears, which routes the
  relu into the matmul program config. Bit-identical (relu commutes with bf16
  rounding), verified PCC 1.0 on 5-sample raw outputs
- Wall median 66.4 -> 65.2 ms (15.34 FPS); full val bit-identical to the fold build
- Width-fusion survey alongside this: QKV and weights_fc were already fused;
  the encoder (unequal 128/32/32/64 branches) and head chains are blocked by
  per-branch LayerNorm — ttnn.group_norm as a segmented-LN substitute measured
  10x worse than layer_norm (mean err 0.052 vs 0.0044, PCC 0.9983) and was
  rejected

**topk_select: radix-sort top-k for the instance bank** (branch perf/topk-kernel)

- ttnn.topk tile-pads the [1, 900] confidence row to 32 x 928 and bitonic-sorts
  all 32 rows on ONE core — 97% of its work is tile padding. The two calls
  (top-300 merge, top-600 cache) cost 1.89 + 2.76 ms/frame
- New op reads only the real row (two 32 B face-row reads per tile, no untilize
  op) and sorts with a 2-pass LSD radix over bf16 bits mapped to monotonic
  uint16 keys — pure integer work, immune to the FPU-less-core soft-float trap.
  Emits uint32 indices directly, absorbing the typecast that followed
- 1893/2759 -> 161 us/call. Device kernel total 65.93 -> 61.94 ms/frame; wall
  median 68.6 -> 67.2 ms (14.87 FPS) — the wall keeps ~2.6 ms less than the
  device saving because the frame is now partly host-bound
- Ordering: descending, ties to the LOWER index (torch semantics, verified
  16/16 against a stable reference incl. 100-way ties). ttnn.topk leaves tie
  order unspecified, so this is not bit-identical to the fallback on tied
  confidences; full val gates the swap. TT_TOPK_KERNEL=0 restores ttnn.topk

**`a5728ad` — row-major FPN 3x3 conv IO to skip conv2d's internal reshapes**

- The FPN 3x3 convs fall into conv2d's DRAM-slice mode, which reshapes its input
  flat -> NHWC and its output NHWC -> flat internally. On TILE tensors both moves are
  real (885 + 868 us at level 0 alone); on ROW_MAJOR with the channel dim unchanged
  they are free views
- Two lines: untilize the conv input, `output_layout=ROW_MAJOR` on the conv. The RM
  output also suits the consumer — DFA reads these maps as RM NHWC for grid_sample
- ReshapeView 5.34 -> 3.38 ms/frame, no tilize/untilize growth. Device kernel total
  68.63 -> 66.61 ms/frame, wall median 71.7 -> 69.7 ms. Full val bit-identical
  (mAP 0.4487746)

**`e39a78c` — block-diagonal weights_fc emits folded attention rows directly**

- The camera-embed linear produced `[N*cams, 416]` with the camera in the row axis;
  folding it to the `[N, cams*416]` row the softmax needs was a TILE reshape moving
  every element, 139 us x 6 layers = 0.81 ms/frame
- Instead the operands are built wide — rows `[f|f|f] + [c0|c1|c2]` — against a
  block-diagonal copy of the weight, so the linear emits the folded row itself.
  Same bytes materialised, and 3x the MACs priced at zero: the matmul measured
  75.7 -> 71.4 us/call because setup, not FLOPs, dominates at this size
- Net -0.68 ms/frame (tilize of the wider rows gives back 0.48 of the 1.16 saved).
  Device kernel total 66.61 -> 65.93 ms/frame, wall median 68.6 ms (14.59 FPS over
  4250 samples). Full val bit-identical — off-diagonal zeros add exact 0.0 into the
  fp32 accumulator

**`93f3f3f` — parallelise `grid_compact`, store the grid as Q14 fixed point**

- Was 1 core. Split across 63: each core tests its own row slice, writes one mask byte per
  row, relays its block to core 0 over the NOC and bumps a semaphore; core 0 walks the mask
  and emits the kept rows
- **762.9 -> 189.8 us/call, 4.58 -> 1.14 ms/frame**, bit-identical on 10 cases
- Grid stored Q14 in a `UINT16` container: coordinates are clamped to [-2, 2], so bf16 spent
  an exponent on a range that never varies. mAP 0.4000 -> 0.4008

**`284f1d7` — keep the attention weights in the compact `[N, CLP*G]` layout**

- The linear already produces that layout; it was being reshaped and transposed into
  `[N, CLP, G]` for the softmax and the fusion
- ReshapeView 11.84 -> 8.39, Transpose 1.90 -> 0.10, Softmax 0.83 -> 0.00 ms/frame
- Net **-4.39 ms/frame**

**`1a3a707` / `de99ca8` — kps_project_fused on the tile path, then its divide on the SFPU**

The op was 17.4 ms of a 91 ms frame, the single largest, and ran entirely on dataflow cores
that have no FPU. Two stages:

- *Tile path.* A tile program factory already existed but was unreachable — three things had
  drifted from the scalar path and would have produced silently wrong output: the writer
  emitted f32 into a Q14 `UINT16` page, `out_page_size` hardcoded `sizeof(float)` and
  overwrote the next anchor's row, and the reader read the anchor as bf16 after the head
  started storing it fp32. Fixed, the rotation and projection run as FPU matmuls:
  **90.56 -> 81.67 ms/frame**
- *SFPU divide.* The writer still did a perspective divide, normalise, clamp and Q14 convert
  per point in soft float. A probe that skipped exactly that chain priced it at **73% of the
  op**. The reader now reorders `P` so depth is in column 0 — the only column
  `mul_tiles_bcast_cols` can broadcast — and pre-scales the x/y rows by `2*2^14/W` and
  `2*2^14/H`, folding the normalise and the Q14 conversion into the matmul at no extra
  matmul. The compute kernel floors depth against a constant tile, takes the reciprocal on
  the SFPU, and multiplies by the broadcast. **kps 19.61 -> 3.85 ms/frame, e2e 81.67 ->
  77.7 ms**

Two things were deliberately left alone, both for numerical reasons:

- The `-1` offset is not folded into the matrix. It would make the numerator `s*px - o*pz`,
  a difference of two nearly equal numbers wherever a point lands near the image centre, and
  this path's grid error is already 100x the scalar one. Subtracting it in the writer is one
  instruction
- The depth floor (`max(pz, 1e-5)`) is not optional and cost a debugging round. Without it a
  point behind the camera has `pz <= 0`, the reciprocal is negative, and `px/pz` returns
  sign-flipped at a plausible magnitude — an in-bounds sample of the wrong pixel rather than
  an out-of-bounds one. Grid PCC does not catch it, because that metric compares in-bounds
  points only; the DFA output does, at 0.789

Accuracy across the two stages: mAP 0.44756 -> 0.44733 -> **0.44802**, i.e. unchanged. The
cost is localisation only — **mATE 0.5532 -> 0.5620** — because the projection is now bf16
tile matmuls rather than software fp32. `TT_KPS_TILE=0` restores the fp32 path.

**`fe6c89a` — materialise the camera-embed add instead of broadcasting it**

- The broadcast form put 1 and 3 in the tile axis, which pads to 32: a 1.3 MB result
  computed through a 14 MB intermediate, 91% padding. reshape + add + reshape = 3.93
  ms/frame
- repeat_interleave + repeat + plain add: every axis tile-aligned, 882 -> 515 us/call
- Full val bit-identical (same pairs summed in the same order); 77.7 -> 75.4 ms

**`5cbe641` / `2c9f9f9` — grid_precompute: grid_sample's coordinate math on Tensix**

- The sampler's reader derived pixel indices and bilinear weights per point in soft float
  on an FPU-less core — 62% of the op. A new op computes the 6-field precomputed form
  after compaction (SFPU for all values, FPU only for 0/1 selector routing) and
  grid_sample consumes it with `use_precomputed_grid=True`
- The op itself took four measured rounds to get cheap (948 -> 518 us/call): resident
  constants (68 DMA barriers -> 1), hardware format conversion (packer/SFPU instead of
  soft-float casts, was 682 us), removing a `c % 6` these dividerless cores turned into a
  library call, and FIELD-MAJOR output rows so the writer copies contiguous runs
- Full val: mAP 0.44802 -> **0.44877**, DFA PCC identical at 0.999830 either way.
  Latency 75.63 -> **71.66 ms** (13.96 FPS). `TT_GRID_PRECOMP=0` restores the reader math
- What remains in the op is mostly per-call fixed cost: kernel 251 us vs FW+dispatch 267
  us, so further kernel work is capped at ~0.6 ms/frame and was left on the table

### Rejected by measurement

Recorded so they are not retried.

**Precision knobs.** Baseline mAP 0.4311, 1500 val samples.

| Setting | Effect on mAP |
| :--- | :---: |
| `MathFidelity.HiFi2` | -0.024 |
| `fp32_dest_acc_en=True` | -0.005 |
| `HiFi4` + `fp32_dest_acc_en=True` | -0.009 |

- All three raise mAOE while lowering mAP
- The earlier diagnosis that the accuracy gap came from math fidelity and accumulator width
  was wrong: no precision knob repairs a wrong formula
- Simulating "every intermediate is bf16" in PyTorch (`TorchFunctionMode`) reaches PCC
  0.9928 against the 0.907 that needed explaining — bf16 storage was never the cause

**A device-side precomputed-grid emitter.** This one is worth reading before anyone tries
it again, because the lever it chases is real and the arithmetic still does not close.

`grid_sample` accepts a precomputed grid: 6 fields per point (pixel coordinates and the four
bilinear weights) instead of 2 normalised coordinates, which removes the ~928 of ~1494
cycles its reader spends per point turning one into the other on an FPU-less core. Measured
on this model's shapes it is **2.78x** — consistent across all four FPN levels, 10.22 ->
3.68 ms/frame, so **-6.5 ms** is genuinely on the table.

The obstacle is that `ttnn.prepare_grid_sample_grid` runs on the host, and this model's grid
is produced on device. Emitting the format from a kernel instead does not remove the work,
it relocates it: the grid is per (point, LEVEL), so four grids are needed and the total is
unchanged. Priced by adding exactly that math to `grid_compact` — which already holds every
row in L1 for the bounds test, across 63 cores — and discarding the result: **+38.5
ms/frame**, six times the prize. Even computing only the rows compaction keeps (25-43%) puts
it at 9.6-16.6 ms, still above 6.5.

The only surviving variant is doing it on the SFPU, where `kps_project_fused` proved the FPU
is worth roughly 2x for comparable math. That needs the grid in tiles, and it is ROW_MAJOR
int16 — the relayout this stack punishes hardest (width-1 assembly measured 71 ms against
0.4 ms field-major, 178x). Not attempted.

`use_precomputed_grid` itself is upstream, costs nothing when false (`if constexpr`), and is
left alone. What is closed is the device-side producer, not the feature.

---

## v2

Full val: mAP 0.3974, NDS 0.5190. Latency 95.2 ms/sample (single scene).

| Change | Measured effect |
| :--- | :--- |
| **`74b1c7e`** 1 KB upload pages — a ROW_MAJOR row is one page, so 4-channel rows made h2d 540,672 transfers of 8 B against 0.13 ms of actual data | h2d 41.0 -> 0.55 ms |
| **`199c2f7`** Pooled OOB compaction — drop anchors that project outside every camera before `grid_sample` sees them, pooling survivors across cameras so the fixed budget covers the busiest *frame* rather than the busiest *camera* | 103.7 -> 95.2 ms, bit-identical |
| **`659e356`** `grouped_weighted_sum` RM mode inherited the previous op's compute-pipeline configuration, so it was wrong on its first call after any other op — silently, and only in that mode | mAP 0.3933 -> 0.3968, no speed cost |
