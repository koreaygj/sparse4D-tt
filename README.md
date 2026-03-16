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


## To-do

- [ ] model module by module convert by tt-nn
  - [x] ResNet50
  - [x] FPN Neck
  - [x] DeformableFeature Aggregation
  - [x] MultiheadAttention
  - [x] AsymmetricFFN
  - [ ] Instance Bank
  - [ ] Validation
- [ ] Integration
  - [ ] varify by test dataset
  - [ ] .ttnn flatbuffer compile
  - [ ] get fps matric
- [ ] Optimization
  - [ ] kernel fusion
  - [ ] Deformable Aggregation
- [ ] Advanced feature
  - [ ] ResNet pcc improvement use fold


## Model Architecture

```
Input (6 cameras, 704x256)
    │
    ▼
┌──────────────────────┐
│   ResNet50 Backbone   │   pretrained, 4 stages
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│      FPN Neck         │   4 levels (stride 4/8/16/32), out_channels=256
└──────────┬───────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│                    Sparse4DHead                       │
│                                                       │
│   Decoder Layer x6 (1 single-frame + 5 multi-frame)  │
│   ┌─────────────────────────────────────────────┐     │
│   │  [temp_gnn] → [gnn] → [norm]               │     │
│   │  → [DeformableFeatureAggregation]           │     │
│   │  → [FFN] → [norm] → [refine]               │     │
│   └─────────────────────────────────────────────┘     │
│                                                       │
│   Instance Bank (900 anchors, 600 temporal)           │
└──────────────────────┬───────────────────────────────┘
                       ▼
              3D Bounding Boxes
              (position, size, rotation, velocity)
```

### Key Modules

| Module | Description | Primary Operations |
|--------|-------------|--------------------|
| ResNet50 | Image feature extraction | Conv2d, BatchNorm, ReLU |
| FPN | Multi-scale feature fusion | Conv2d, Upsample (nearest 2x) |
| DeformableFeatureAggregation | Multi-view feature sampling & aggregation | grid_sample, matmul, softmax, weighted sum |
| MultiheadAttention | Inter-instance relation modeling | matmul, softmax |
| AsymmetricFFN | Feed-forward network | Linear, ReLU, LayerNorm |
| Instance Bank | Temporal instance management | concat, indexing |

### Deformable Feature Aggregation

The core operation of this model, with two execution paths:

```python
# Path 1: Custom CUDA kernel (use_deformable_func=True)
# - Single fused kernel for bilinear sampling + weighted aggregation
# - 1 memory access pass, best performance
features = DAF(*feature_maps, points_2d, weights)

# Path 2: PyTorch fallback (use_deformable_func=False)
# - F.grid_sample called per level (4 times) for bilinear sampling
# - Separate weighted sum operations
# - Mathematically identical results
for fm in feature_maps:  # 4 levels
    features.append(F.grid_sample(fm.flatten(end_dim=1), points_2d))
features = weights[..., None] * features
features = features.sum(dim=2).sum(dim=2)
```

**For Tenstorrent porting, we use Path 2 (fallback).** All operations are composed of standard PyTorch ops, making TT-NN conversion possible.

## Tenstorrent Software Stack

The Tenstorrent software stack used for porting has a 3-layer architecture:

```
┌─────────────────────────────────────────────┐
│            TT-Forge (Top Layer)              │
│   MLIR-based compiler                        │
│   PyTorch/JAX/TF models → auto compilation   │
├─────────────────────────────────────────────┤
│            TT-NN (Middle Layer)              │
│   PyTorch-style operator library (200+ ops)  │
│   Python/C++ API                             │
├─────────────────────────────────────────────┤
│         TT-Metalium (Bottom Layer)           │
│   Direct hardware access SDK                 │
│   Custom C++ kernel development              │
└─────────────────────────────────────────────┘
```

### Inference Pipeline

```
PyTorch model (.pt)
    ↓  Rewrite model with TT-NN
TT-NN model (Python)
    ↓  TT-Forge compilation
MLIR intermediate representation (.mlir)
    ↓  ttmlir-translate
Flatbuffer binary (.ttnn)
    ↓  ttrt runtime
Execution on Tenstorrent device
```

| File Format | Purpose |
|-------------|---------|
| `.ttsys` | Hardware system descriptor (target device info) |
| `.ttnn` | Compiled execution binary (for inference) |
| `.mlir` | Intermediate representation (compilation stage) |

## Operator Compatibility

With `use_deformable_func=False`, every PyTorch operation in the model has a corresponding TT-NN operator.

### Compatibility Table

| PyTorch | TT-NN | Status | Notes |
|---------|-------|:------:|-------|
| `F.grid_sample` | `ttnn.grid_sample` | Supported | Channel must be tile-aligned, NHWC format, zeros padding only |
| `nn.Conv2d` | `ttnn.conv2d` | Supported | |
| `nn.BatchNorm2d` | `ttnn.batch_norm` | Supported | Must be tilized, interleaved, rank 4 |
| `nn.ReLU` | `ttnn.relu` | Supported | |
| `nn.Linear` | `ttnn.linear` | Supported | Activation fusion available |
| `nn.LayerNorm` | `ttnn.layer_norm` | Supported | |
| `F.gelu` | `ttnn.gelu` | Supported | |
| `softmax` | `ttnn.softmax` | Supported | |
| `matmul` / `@` | `ttnn.matmul` | Supported | |
| `F.interpolate` | `ttnn.upsample` | Supported | nearest 2x (used in FPN) |
| `torch.cat` | `ttnn.concat` | Supported | |
| `reshape / permute` | `ttnn.reshape` / `ttnn.permute` | Supported | |
| `torch.sum` | `ttnn.sum` | Supported | |
| `torch.clamp` | `ttnn.clamp` | Supported | TILE layout required |
| `torch.exp` | `ttnn.exp` | Supported | |
| `nn.Dropout` | - | N/A | No-op during inference, removed |

### grid_sample Constraints

The `grid_sample` used in the Deformable Feature Aggregation fallback path is supported in TT-NN with the following constraints:

- **Input format**: `(N, H, W, C)` — differs from PyTorch's `(N, C, H, W)`, permute required
- **Channel alignment**: `channel_size % tile_width == 0` required (embed_dims=256 satisfies this)
- **Padding mode**: only `"zeros"` supported
- **Channel size**: 256 or less recommended ([GitHub #28513](https://github.com/tenstorrent/tt-metal/issues/28513))

## Porting Strategy

### Phase 1: Switch to Fallback Path

Remove custom CUDA operation dependency.

```python
# Change in config
use_deformable_func = False  # Disable CUDA DAF → use PyTorch grid_sample fallback
```

This makes the entire model composed of standard PyTorch operations.

### Phase 2: Module-by-Module TT-NN Conversion

Follow the 3-step procedure from the [PyTorch → TT-NN conversion guide](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/converting_torch_model_to_ttnn.html):

1. **Rewrite nn.Module → functional PyTorch**
2. **Replace PyTorch ops → ttnn ops**
3. **Optimize dtype / memory layout**

Module conversion order:

```
1. ResNet50 backbone     ← Official TT-NN demo exists, verified
2. FPN neck              ← Conv2d + nearest upsample
3. DeformableFeatureAggregation (fallback)
                         ← grid_sample + matmul + softmax + sum
4. MultiheadAttention    ← matmul + softmax
5. AsymmetricFFN         ← Linear + ReLU + LayerNorm
6. Instance Bank         ← concat + indexing
```

Key conversion patterns:

```python
# Before (PyTorch)
output = self.output_proj(features)
weights = self.weights_fc(feature).softmax(dim=-2)

# After (TT-NN)
output = ttnn.linear(features, parameters.output_proj.weight, bias=parameters.output_proj.bias)
weights = ttnn.softmax(ttnn.linear(feature, parameters.weights_fc.weight, bias=parameters.weights_fc.bias), dim=-2)
```

NCHW → NHWC conversion points:

```python
# Layout conversion at Conv/grid_sample boundaries
feature = ttnn.permute(feature, (0, 2, 3, 1))  # NCHW → NHWC
```

### Phase 3: (Optional) Custom Kernel Optimization

If profiling reveals Deformable Feature Aggregation as a bottleneck, implement a fused kernel using TT-Metalium.

```
Fallback (Phase 2)                    Fused Kernel (Phase 3)
──────────────────                    ──────────────────────
grid_sample x4 (per level)   →       Single kernel
  ↓ intermediate tensor store          bilinear sampling
multiply (weights)                     + weighted aggregation
  ↓ intermediate tensor store          = 1 memory access pass
sum (cam) → sum (level)
= multiple memory accesses
```

Custom op procedure: [Adding New TT-NN Operation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html)

### Phase 4: Integration & Deployment

```bash
# Generate system descriptor
ttrt query --save-artifacts

# Compile MLIR → Flatbuffer
ttmlir-translate --ttnn-to-flatbuffer model.mlir -o sparse4d.ttnn

# Run inference
ttrt run sparse4d.ttnn
```

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

```bash
# PyTorch baseline inference (for validation)
cd Sparse4D
python tools/test.py projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py <checkpoint> --eval bbox

# TT-NN model inference
# (To be added after Phase 2 completion)
```

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
