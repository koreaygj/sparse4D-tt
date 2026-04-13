# Installation Guide

## Prerequisites

- Tenstorrent Wormhole N300 (2-chip) or compatible hardware
- TT-Metalium SDK installed ([Getting Started](https://docs.tenstorrent.com/getting-started/README.html))
- Python 3.10+
- nuScenes dataset ([download](https://www.nuscenes.org/download))

## 1. Clone & Dependencies

```bash
git clone <repository-url>
cd sparse4D-tt-performance
pip install -r requirement.txt
```

## 2. Custom Kernel Build

This project uses 3 custom TT-Metal kernels. They must be built into the tt-metal library before running.

### 2.1 Copy kernel source

```bash
cp -r tt-metal/ops/kps_project_fused ~/tt-metal/ttnn/cpp/ttnn/operations/pool/
cp -r tt-metal/ops/grouped_weighted_sum ~/tt-metal/ttnn/cpp/ttnn/operations/pool/
cp -r tt-metal/ops/transposed_s2i ~/tt-metal/ttnn/cpp/ttnn/operations/pool/
```

### 2.2 Register in CMake

**File: `~/tt-metal/ttnn/cpp/ttnn/operations/pool/CMakeLists.txt`**

Add to `file(GLOB_RECURSE kernels ...)`:
```cmake
    kps_project_fused/device/kernels/*
    grouped_weighted_sum/device/kernels/*
    transposed_s2i/device/kernels/*
```

Add to `target_sources(ttnn_op_pool PRIVATE ...)`:
```cmake
    kps_project_fused/kps_project_fused.cpp
    kps_project_fused/device/kps_project_fused_device_operation.cpp
    kps_project_fused/device/kps_project_fused_program_factory.cpp
    grouped_weighted_sum/grouped_weighted_sum.cpp
    grouped_weighted_sum/device/grouped_weighted_sum_device_operation.cpp
    grouped_weighted_sum/device/grouped_weighted_sum_program_factory.cpp
    transposed_s2i/transposed_s2i.cpp
    transposed_s2i/device/transposed_s2i_device_operation.cpp
    transposed_s2i/device/transposed_s2i_program_factory.cpp
```

### 2.3 Register nanobind

**File: `~/tt-metal/ttnn/CMakeLists.txt`**

Add to the nanobind source list (near other `pool/` entries):
```cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/kps_project_fused/kps_project_fused_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/grouped_weighted_sum/grouped_weighted_sum_nanobind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ttnn/operations/pool/transposed_s2i/transposed_s2i_nanobind.cpp
```

### 2.4 Register Python bindings

**File: `~/tt-metal/ttnn/cpp/ttnn-nanobind/__init__.cpp`**

Add includes (near other `pool/` includes):
```cpp
#include "ttnn/operations/pool/kps_project_fused/kps_project_fused_nanobind.hpp"
#include "ttnn/operations/pool/grouped_weighted_sum/grouped_weighted_sum_nanobind.hpp"
#include "ttnn/operations/pool/transposed_s2i/transposed_s2i_nanobind.hpp"
```

Add bind calls in the `m_pool` section (near `grid_sample::bind_grid_sample(m_pool)`):
```cpp
    kps_project_fused::bind_kps_project_fused(m_pool);
    grouped_weighted_sum::bind_grouped_weighted_sum(m_pool);
    transposed_s2i::bind_transposed_s2i(m_pool);
```

### 2.5 Build & Install

```bash
cd ~/tt-metal
cmake --build build_Release --target ttnn -j$(nproc)

# Copy built libraries to Python import path
cp build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
cp build_Release/ttnn/_ttnncpp.so ttnn/ttnn/_ttnncpp.so
cp build_Release/tt_metal/libtt_metal.so tt_metal/libtt_metal.so
```

### 2.6 Verify

```bash
python -c "import ttnn; print(ttnn.kps_project_fused); print(ttnn.transposed_s2i); print(ttnn.grouped_weighted_sum)"
```

## 3. Checkpoint

Place checkpoint files in `ckpt/`:
```
ckpt/
  latest.pth        # FP32 trained weights
  bf16_latest.pth   # BF16 QAT finetuned weights (optional)
```

## 4. nuScenes Data

```
nuscenes/trainval/          # or symlink to dataset path
nuscenes_anno_pkls/
  nuscenes_infos_val.pkl    # validation info pickle
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ethernet timeout` | `tt-smi -r 0,1` to reset devices |
| `run_mailbox` error | Device hang — reset with `tt-smi -r 0,1` |
| Kernel not found | Clear JIT cache: `rm -rf ~/.cache/tt-metal-cache` |
| `LD_PRELOAD` needed | `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmpi_cxx.so.40.30.1:/usr/lib/x86_64-linux-gnu/libmpi.so.40` |
| `reshape different volumes` | Library version mismatch — rebuild and copy libraries |

## Custom Kernels Overview

| Kernel | Python API | Role |
|--------|-----------|------|
| `kps_project_fused` | `ttnn.kps_project_fused(...)` | 3D keypoint rotation + projection to 2D in single dispatch |
| `transposed_s2i` | `ttnn.transposed_s2i(...)` | Rearrange L1 sharded grid_sample output to [CLP, N, C] DRAM layout |
| `grouped_weighted_sum` | `ttnn.grouped_weighted_sum(...)` | TILE-based grouped weighted sum with mul_bcast_cols + L1 accumulation |
