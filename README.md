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
pip install -r Sparse4D/requirement.txt
```

### Running

**nuScenes Val Evaluation**

```bash
# TT-NN inference (dual-device, full val set)
TT_METAL_LOGGER_LEVEL=ERROR python test/sparse4d_nuscenes_val.py \
  --data-root /path/to/nuscenes/trainval \
  --dual-device

# With options
python test/sparse4d_nuscenes_val.py \
  --data-root /path/to/nuscenes/trainval \
  --dual-device \
  --num-samples 100 \          # quick test (default: all 6019)
  --bf16 \                     # use BF16 QAT checkpoint
  --fidelity hifi4 \           # override math fidelity (lofi/hifi2/hifi4)
  --grid-sample-lerp           # use grid_sample_lerp instead of grid_sample

# PyTorch baseline (for comparison)
cd Sparse4D
python nuscenes_val.py --data-root /path/to/nuscenes/trainval
```

**Inference Video**

```bash
# Generate detection video from nuScenes samples
TT_METAL_LOGGER_LEVEL=ERROR python run_inference_video.py \
  --data-root /path/to/nuscenes/trainval \
  --dual-device \
  --frames 80 \
  --conf 0.3 \
  --output inference_video_tt.mp4

# FPS benchmark mode (no video output)
python run_inference_video.py --dual-device --mode fps --frames 50
```

## Compare with PyTorch

### matrics

|           | **PyTorch (CUDA)** | **TT-NN (N300)** | **Gap** |
| --------- | ------------------ | ---------------- | ------- |
| **mAP**   | 0.4529             | 0.3975           | -0.055  |
| **NDS**   | 0.5602             | 0.5179           | -0.042  |
| mATE      | 0.5455             | 0.6196           | +0.074  |
| mASE      | 0.2622             | 0.2693           | +0.007  |
| mAOE      | 0.4373             | 0.4777           | +0.040  |
| mAVE      | 0.2195             | 0.2658           | +0.046  |
| mAAE      | 0.1987             | 0.1765           | -0.022  |
| **Speed** | ~120ms (A100)      | **330ms**        | -       |


### Inference video

https://github.com/user-attachments/assets/3752d051-9e4a-4a9d-9410-eec0bdcb3027

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
