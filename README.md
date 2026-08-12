# Sparse4D on Tenstorrent

Sparse4D v3 3D object detection, ported to Tenstorrent (Wormhole/Blackhole).

## Overview

| Item | Details |
|------|---------|
| Source Model | [Sparse4D v3](https://github.com/HorizonRobotics/Sparse4D) (ICCV 2023) |
| Framework | PyTorch + MMDetection3D |
| Dataset | nuScenes (10 classes, 6 cameras) |
| Target Hardware | Tenstorrent Wormhole / Blackhole |
| Input Resolution | 704 x 256, 6 cameras |
| Baseline Performance | NDS 0.5637, mAP 0.4647 (ResNet50) |


## Performance

### Speed

|  | PyTorch | TT-NN Pure (N300) | ~~v1~~ | ~~v2~~ | **Custom Kernels v3 (N300)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Latency / sample** | ~95 ms | 235 ms | ~~122 ms~~ | ~~95.2 ms~~ | **77.7 ms** |
| **FPS** | 10.5 | 4.2 | ~~8.2~~ | ~~10.51~~ | **12.86** |

- Metric: `model.forward` on one 6-camera sample
- v3: median over a full 6019-sample val run — mean 78.1, p90 78.6, max 349.4
- Scene choice matters: anchors surviving the OOB compaction vary by scene, and one scene
  replayed reads 99.9 ms against 90.7 ms across scenes on the same build, so a short bench
  on one scene is not the number
- v1/v2 were taken on a single scene — history, not a controlled baseline

Build modes:

- **Pure TT-NN** — stock ttnn ops only, no custom build; automatic fallback
- **Custom Kernels** (recommended) — 4 Metalium kernels (`kps_project_fused`,
  `transposed_s2i`, `grouped_weighted_sum`, `grid_compact`) + a patch to upstream
  `grid_sample`; see [docs/INSTALL.md](docs/INSTALL.md)

Per-change measurements: **[docs/CHANGELOG.md](docs/CHANGELOG.md)**

### Accuracy

Full nuScenes val, 6019 samples.

|  | PyTorch (CUDA) | ~~TT-NN v2~~ | **TT-NN v3 (N300)** | Gap (v3) |
| :--- | :---: | :---: | :---: | :---: |
| **mAP** | 0.4529 | ~~0.3974~~ | **0.4480** | -0.0049 |
| **NDS** | 0.5602 | ~~0.5190~~ | **0.5520** | -0.0082 |
| **mATE** | 0.5455 | ~~0.6173~~ | **0.5620** | +0.0165 |
| **mASE** | 0.2622 | ~~0.2693~~ | **0.2631** | +0.0009 |
| **mAOE** | 0.4373 | ~~0.4730~~ | **0.4691** | +0.0318 |
| **mAVE** | 0.2195 | ~~0.2624~~ | **0.2176** | -0.0019 |
| **mAAE** | 0.1987 | ~~0.1747~~ | **0.2080** | +0.0093 |

Comparability:

- Controlled pair either side of the softmax fix — **0.4019 -> 0.4476 mAP**, same
  checkpoint (`sparse4dv3_r50.pth`), same tree, one changed function. That fix is where
  essentially all of the accuracy came from
- The remaining 0.4476 -> 0.4480 is the tile/SFPU projection path, measured across two more
  full vals and inside noise either way. What it does move is **mATE, 0.5532 -> 0.5620** —
  the projection runs as bf16 tile matmuls rather than software fp32, and localisation is
  the only metric that touches. `TT_KPS_TILE=0` buys that back for 12 ms a frame
- v2 column: several changes older, checkpoint not recorded — history
- PyTorch column: carried from earlier in this file, not re-run here

**Root cause — the softmax normalised over the wrong set.**

- SPMD gives each device 3 of 6 cameras, so its sampling-point axis is 156 of 312
- `_softmax_clp` built the denominator from its own 156
- Each device's weights summed to 1 alone — row sums 1.0063 each, 2.01 together
- So the post-fusion `all_reduce` added two complete distributions

**Fix** — reduce the denominator across devices, and the max shift with it.

- `exp(l - m0)` and `exp(l - m1)` are not summable — both devices must subtract the same
  constant; any common constant is exact
- `ttnn.all_reduce` is Sum-only, so it uses the mean of the per-device maxima
- Both tensors are per-anchor — the collectives cost nothing measurable
- DFA output PCC 0.9883 -> 0.9999, identical on the pure-TT-NN fallback path

**Correction** — this section previously blamed `MathFidelity` and `fp32_dest_acc_en`. The
hardware claims were right, the diagnosis was not: all three settings **cost** mAP when
measured. No precision knob repairs a wrong formula. Figures in
[docs/CHANGELOG.md](docs/CHANGELOG.md#rejected-by-measurement).

Cleared by measurement, and still clear:

- backbone/FPN 0.9998 vs PyTorch fp32
- sampling grid 0.999997, sampled features 0.999975
- compaction mask lossless — 0 of 223,496 zeroed rows has a non-zero feature
- gws bf16 accumulator 0.99994

### Inference video

https://github.com/user-attachments/assets/fc16cb97-e5b9-454b-8ce0-ea83a270ca95


## Setup

### Requirements

- Tenstorrent hardware (Wormhole or Blackhole)
- TT-Metalium SDK
- TT-NN
- Python 3.8+
- PyTorch 2.0+

### Installation

```bash
# 1. Install Tenstorrent software stack
# See https://docs.tenstorrent.com/getting-started/README.html

# 2. Clone project
git clone <repository-url>
cd project

# 3. Install dependencies
pip install -r Sparse4D-tt/requirement.txt
```

### tt-metal Custom Kernel Build

- 4 custom kernels — `kps_project_fused`, `grouped_weighted_sum`, `transposed_s2i`,
  `grid_compact`
- 1 patch to upstream `grid_sample`
- All must be built into the tt-metal library — see **[docs/INSTALL.md](docs/INSTALL.md)**

### Running

**nuScenes Val Evaluation**

```bash
# TT-NN inference (dual-device mesh, full val set)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmpi_cxx.so.40.30.1:/usr/lib/x86_64-linux-gnu/libmpi.so.40
TT_METAL_LOGGER_LEVEL=ERROR python test/sparse4d_nuscenes_val.py \
  --data-root /path/to/nuscenes/trainval \
  --dual-device

# With options
TT_METAL_LOGGER_LEVEL=ERROR python test/sparse4d_nuscenes_val.py \
  --data-root /path/to/nuscenes/trainval \
  --dual-device \
  --num-sample 100 \           # quick test (default: all 6019)
  --bf16 \                     # use BF16 checkpoint (ckpt/bf16_latest.pth)
  --fidelity hifi4             # override math fidelity (lofi/hifi2/hifi4)

# PyTorch baseline (for comparison)
cd Sparse4D
conda activate sparse4d
python nuscenes_val.py --data-root /path/to/nuscenes/trainval
```

**Inference Speed Profiling**

```bash
TT_METAL_LOGGER_LEVEL=ERROR python debug/profile_inference.py
```

**Inference Video**

```bash
# Generate detection video from nuScenes samples
TT_METAL_LOGGER_LEVEL=ERROR python tools/run_inference_video.py \
  --data-root /path/to/nuscenes/trainval \
  --dual-device \
  --frames 80 \
  --conf 0.3 \
  --output inference_video_tt.mp4

# FPS benchmark mode (no video output)
TT_METAL_LOGGER_LEVEL=ERROR python tools/run_inference_video.py \
  --dual-device --mode fps --frames 50
```

> **Note**: `LD_PRELOAD` is required for mesh CCL (all_reduce) on some systems. If you get Ethernet timeout errors, run `tt-smi -r 0,1` to reset devices.

## References

### Sparse4D

- [Sparse4D v3 Paper](https://arxiv.org/abs/2311.11722) — Advancing End-to-End 3D Detection and Tracking
- [Sparse4D GitHub](https://github.com/HorizonRobotics/Sparse4D)

### Tenstorrent Official Documentation

- [Software Stack Overview](https://docs.tenstorrent.com/getting-started/tt-software-stack.html)
- [Installation Guide](https://docs.tenstorrent.com/getting-started/README.html)
- [PyTorch → TT-NN Conversion Guide](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/converting_torch_model_to_ttnn.html)
- [TT-NN API Reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html)
- [ttnn.grid_sample](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.grid_sample.html)
- [Adding New TT-NN Operation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html)
- [TT-Metalium Getting Started](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/get_started/get_started.html)
- [TTRT Runtime](https://docs.tenstorrent.com/tt-mlir/ttrt.html)
- [Flatbuffer Format](https://docs.tenstorrent.com/tt-mlir/flatbuffers.html)

### GitHub Repositories

- [tt-metal (TT-NN + TT-Metalium)](https://github.com/tenstorrent/tt-metal)
- [tt-forge (MLIR Compiler)](https://github.com/tenstorrent/tt-forge)
- [tt-forge-onnx](https://github.com/tenstorrent/tt-forge-onnx)

### Related Issues

- [grid_sample generality — GitHub #28513](https://github.com/tenstorrent/tt-metal/issues/28513)
- [grid_sample performance — GitHub #27904](https://github.com/tenstorrent/tt-metal/issues/27904)
- [Deformable conv/attention support request — GitHub #17076](https://github.com/tenstorrent/tt-metal/issues/17076)
