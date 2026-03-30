"""
Sparse4D Full TT-NN Inference on nuScenes

Runs the complete pipeline: Image → ResNet → FPN → Sparse4DHead → 3D BBoxes
on Tenstorrent hardware.

Usage:
  python test/sparse4d_inference.py
  python test/sparse4d_inference.py --ckpt work_dirs/.../latest.pth
  python test/sparse4d_inference.py --num-samples 10
"""

import argparse
import os
import sys
import time

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.expanduser("~/project/tt-metal")
)
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/project/Sparse4D"))

import torch
import numpy as np
import ttnn

from model.sparse4d import Sparse4DInference


# nuScenes class names
CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def load_pytorch_model(ckpt_path: str):
    """Load full Sparse4D PyTorch model from checkpoint."""
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg_path = os.path.expanduser(
        "~/project/Sparse4D/projects/configs/"
        "sparse4dv3_temporal_r50_1x8_bs6_256x704.py"
    )
    cfg = Config.fromfile(cfg_path)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print("  WARNING: No checkpoint loaded, using random weights")

    model.eval()
    return model, cfg


def build_dataloader(cfg, num_samples=None):
    """Build nuScenes test dataloader."""
    from mmdet3d.datasets import build_dataset, build_dataloader as _build_dl

    dataset = build_dataset(cfg.data.test)
    if num_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))

    dataloader = _build_dl(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )
    return dataloader


def extract_metas(data: dict) -> dict:
    """Extract metas from mmdet3d data dict for TT-NN Head."""
    img_metas = data["img_metas"][0].data[0]

    # projection_mat: [bs, 6, 4, 4]
    proj = torch.tensor(
        np.stack([m["projection_mat"] for m in img_metas]),
        dtype=torch.float32,
    )

    # image_wh: [bs, 6, 2]
    wh = torch.tensor(
        np.stack([m["image_wh"] for m in img_metas]),
        dtype=torch.float32,
    )

    # timestamp: [bs]
    timestamp = torch.tensor(
        [m["timestamp"] for m in img_metas],
        dtype=torch.float64,
    )

    metas = {
        "projection_mat": proj,
        "image_wh": wh,
        "timestamp": timestamp,
        "img_metas": img_metas,
    }
    return metas


def print_detections(results: list, top_k: int = 20):
    """Print top detections."""
    if not results or len(results) == 0:
        print("  No detections.")
        return

    r = results[0]
    boxes = r["boxes_3d"]
    scores = r["scores_3d"]
    labels = r["labels_3d"]

    if len(scores) == 0:
        print("  No detections above threshold.")
        return

    # Sort by score
    sorted_idx = scores.argsort(descending=True)[:top_k]

    print(f"\n  Top-{min(top_k, len(sorted_idx))} detections:")
    print(f"  {'#':<4} {'Class':<20} {'Score':>8} {'X':>8} {'Y':>8} {'Z':>8}")
    print(f"  {'-'*3:<4} {'-'*19:<20} {'-'*7:>8} {'-'*7:>8} {'-'*7:>8} {'-'*7:>8}")

    for rank, idx in enumerate(sorted_idx):
        i = idx.item()
        cls_name = CLASS_NAMES[labels[i].item()] if labels[i] < len(CLASS_NAMES) else "unknown"
        x, y, z = boxes[i, 0].item(), boxes[i, 1].item(), boxes[i, 2].item()
        score = scores[i].item()
        print(f"  {rank+1:<4} {cls_name:<20} {score:>8.4f} {x:>8.2f} {y:>8.2f} {z:>8.2f}")


def run_pytorch_inference(pt_model, data):
    """Run PyTorch inference for comparison."""
    with torch.no_grad():
        img = data["img"][0].data[0].cuda() if torch.cuda.is_available() else data["img"][0].data[0]
        pt_model = pt_model.cuda() if torch.cuda.is_available() else pt_model
        result = pt_model.simple_test(img, **{k: v for k, v in data.items() if k != "img"})
    return result


def main():
    parser = argparse.ArgumentParser(description="Sparse4D TT-NN Inference")
    parser.add_argument(
        "--ckpt", type=str,
        default=os.path.expanduser(
            "~/project/Sparse4D/work_dirs/"
            "sparse4dv3_temporal_r50_1x8_bs6_256x704/latest.pth"
        ),
        help="Path to Sparse4D checkpoint",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--compare-pytorch", action="store_true",
                        help="Also run PyTorch inference for comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("  Sparse4D Full TT-NN Inference")
    print("=" * 70)

    # 1. Load PyTorch model
    print("\n[1/4] Loading PyTorch model...")
    pt_model, cfg = load_pytorch_model(args.ckpt)

    # 2. Open TT device & build TT-NN model
    print("\n[2/4] Building TT-NN model...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        grid = device.compute_with_storage_grid_size()
        print(f"  Device: {device.arch()}, grid: {grid.x}x{grid.y}")

        tt_model = Sparse4DInference.from_pytorch(device, pt_model)

        # 3. Build dataloader
        print(f"\n[3/4] Loading nuScenes data ({args.num_samples} samples)...")
        dataloader = build_dataloader(cfg, args.num_samples)

        # 4. Run inference
        print(f"\n[4/4] Running inference...")
        tt_model.reset()

        total_time = 0
        for sample_idx, data in enumerate(dataloader):
            img = data["img"][0].data[0]  # [1, 6, 3, 256, 704]
            metas = extract_metas(data)

            print(f"\n  --- Sample {sample_idx + 1}/{args.num_samples} ---")

            # TT-NN inference
            t0 = time.time()
            outputs = tt_model.forward(img, metas, bs=1)
            elapsed = time.time() - t0
            total_time += elapsed

            # Post-process
            results = Sparse4DInference.post_process(
                outputs, score_threshold=args.score_threshold
            )

            print(f"  TT-NN inference: {elapsed:.2f}s")
            print_detections(results)

            # Optional: compare with PyTorch
            if args.compare_pytorch:
                pt_result = run_pytorch_inference(pt_model, data)
                print(f"  PyTorch result: {len(pt_result[0]['img_bbox'])} detections")

        # Summary
        avg_time = total_time / max(1, args.num_samples)
        print(f"\n{'=' * 70}")
        print(f"  Summary:")
        print(f"  Samples processed:  {args.num_samples}")
        print(f"  Total time:         {total_time:.2f}s")
        print(f"  Average per sample: {avg_time:.2f}s")
        print(f"  Average FPS:        {1.0 / avg_time:.2f}" if avg_time > 0 else "  N/A")
        print(f"{'=' * 70}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  FAILED: {str(e)[:300]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
