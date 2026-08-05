# Sparse4D on Tenstorrent

Porting the Sparse4D v3 3D object detection model to Tenstorrent devices (Wormhole/Blackhole).

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

|  | PyTorch | TT-NN Pure (N300) | ~~Custom Kernels v1~~ | **Custom Kernels v2 (N300)** |
| :--- | :---: | :---: | :---: | :---: |
| **Latency / sample** | ~95 ms | 235 ms | ~~122 ms~~ | **95.2 ms** |
| **FPS** | 10.5 | 4.2 | ~~8.2~~ | **10.51** |

Latency is `model.forward` on one 6-camera sample, median of 20 frames after warmup.
At 95.2 ms the accelerator matches the CUDA reference on this model.

- **Pure TT-NN**: Standard ttnn ops only, no custom kernel build required. Automatically used as fallback when custom kernels are not built.
- **Custom Kernels** (recommended): 4 custom Metalium kernels (`kps_project_fused`, `transposed_s2i`, `grouped_weighted_sum`, `grid_compact`) plus a patch to upstream `grid_sample` — see [docs/INSTALL.md](docs/INSTALL.md)

**What changed in v2.** The three items below were each measured on this machine; the v1
figure is the previously published one from an earlier state of the tree, so the two
columns are a before/after, not a controlled A/B of a single change.

| Change | Measured effect |
| :--- | :--- |
| 1 KB upload pages — a ROW_MAJOR row is one page, so 4-channel rows made h2d 540,672 transfers of 8 B against 0.13 ms of actual data | h2d 41.0 -> 0.55 ms |
| Pooled OOB compaction — drop the anchors that project outside every camera before `grid_sample` sees them, pooling the survivors across cameras so the fixed budget covers the busiest *frame* rather than the busiest *camera* | 103.7 -> 95.2 ms, bit-identical output |
| `grouped_weighted_sum` RM mode inherited the previous op's compute-pipeline configuration, so it was wrong on its first call after any other op — silently, and only in that mode | mAP 0.3933 -> 0.3968, no speed cost |

### Accuracy

Full nuScenes val, 6019 samples.

|  | PyTorch (CUDA) | ~~TT-NN v1~~ | **TT-NN v2 (N300)** | Gap (v2) |
| :--- | :---: | :---: | :---: | :---: |
| **mAP** | 0.4529 | ~~0.3968~~ | **0.3974** | -0.0555 |
| **NDS** | 0.5602 | ~~0.5192~~ | **0.5190** | -0.0412 |
| **mATE** | 0.5455 | ~~0.6163~~ | **0.6173** | +0.0718 |
| **mASE** | 0.2622 | ~~0.2689~~ | **0.2693** | +0.0071 |
| **mAOE** | 0.4373 | ~~0.4693~~ | **0.4730** | +0.0357 |
| **mAVE** | 0.2195 | ~~0.2590~~ | **0.2624** | +0.0429 |
| **mAAE** | 0.1987 | ~~0.1790~~ | **0.1747** | -0.0240 |

v2 is faster *and* slightly more accurate. The gain came from fixing
`grouped_weighted_sum`, not from tuning: mAP 0.3933 -> 0.3968 for the fix, and a further
+0.0006 from compaction, which is noise-level and in the favourable direction.

**On the remaining -0.056 mAP:** it is not bf16 storage. PyTorch bf16 keeps a
full-precision multiply and accumulates in fp32; two Wormhole settings break both halves
of that, and both are tunable:

- **Math fidelity.** The multiplier is physically 5b x 7b, and `MathFidelity` sets how many
  passes it takes to consume the inputs. At `LoFi` — one pass, and what the backbone and
  FPN currently use — srcA contributes its hidden bit plus only the top 4 mantissa bits, so
  a bf16 operand is truncated to 5 of its 8 bits. That is coarser than bf16, not equal to it.
- **Accumulator width.** With `fp32_dest_acc_en=False` (the default here, everywhere) the
  FPU accumulates in 16-bit rather than fp32.

Measured directly in this project: the same key-point projection written as ttnn
element-wise ops produced errors quantised to 2^-11 / 2^-10 — the fp16 DEST signature —
costing 0.042 mAP, while the identical arithmetic in a custom kernel doing fp32 soft-float
had a maximum error of 5e-8. Input and output dtypes were fp32 in both cases; only the
accumulator differed.

### Inference video

https://github.com/user-attachments/assets/3752d051-9e4a-4a9d-9410-eec0bdcb3027


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

This project uses 4 custom TT-Metal kernels (`kps_project_fused`, `grouped_weighted_sum`, `transposed_s2i`, `grid_compact`) plus one patch to the upstream `grid_sample`, all of which must be built into the tt-metal library.

See **[docs/INSTALL.md](docs/INSTALL.md)** for detailed build instructions.

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
