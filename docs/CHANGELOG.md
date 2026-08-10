# Update log

Per-change measurements. Figures are from this machine unless stated otherwise; "no change"
means measured and found within noise, not unmeasured.

Headline results live in the [README](../README.md); this file records how they were reached.

---

## v3

Full val, 6019 samples, `sparse4dv3_r50.pth`: mAP 0.4019 -> **0.4476**, NDS 0.5186 ->
**0.5534**, mATE 0.5947 -> **0.5532**. Latency 90.7 ms/sample.

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

### Rejected by measurement

Recorded so they are not retried. Baseline mAP 0.4311, 1500 val samples.

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

---

## v2

Full val: mAP 0.3974, NDS 0.5190. Latency 95.2 ms/sample (single scene).

| Change | Measured effect |
| :--- | :--- |
| **`74b1c7e`** 1 KB upload pages — a ROW_MAJOR row is one page, so 4-channel rows made h2d 540,672 transfers of 8 B against 0.13 ms of actual data | h2d 41.0 -> 0.55 ms |
| **`199c2f7`** Pooled OOB compaction — drop anchors that project outside every camera before `grid_sample` sees them, pooling survivors across cameras so the fixed budget covers the busiest *frame* rather than the busiest *camera* | 103.7 -> 95.2 ms, bit-identical |
| **`659e356`** `grouped_weighted_sum` RM mode inherited the previous op's compute-pipeline configuration, so it was wrong on its first call after any other op — silently, and only in that mode | mAP 0.3933 -> 0.3968, no speed cost |
