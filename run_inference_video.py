"""
Sparse4D TT-NN Inference Video Generator (Mesh SPMD + BF16)
Camera-view visualization with 3D bounding boxes on Tenstorrent N300.

Usage:
  source ~/.tenstorrent-venv/bin/activate
  TT_METAL_LOGGER_LEVEL=ERROR python run_inference_video.py --dual-device
  TT_METAL_LOGGER_LEVEL=ERROR python run_inference_video.py --dual-device --frames 80 --conf 0.25
  TT_METAL_LOGGER_LEVEL=ERROR python run_inference_video.py --mode fps --dual-device --frames 50
"""

import argparse
import copy
import gc
import os
import sys
import time

import cv2
import numpy as np
import torch
import ttnn

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.expanduser("~/project/tt-metal")
)
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.dirname(__file__))

from test.sparse4d_nuscenes_val import (
    CLASS_NAMES,
    NuScenesValLoader,
    load_model,
    OPERATION_ORDER,
    SPATIAL_SHAPES,
)

# Class-specific colors (BGR for OpenCV)
CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 200, 255),
    "bus": (0, 150, 255),
    "trailer": (0, 100, 200),
    "construction_vehicle": (0, 80, 180),
    "pedestrian": (255, 0, 0),
    "motorcycle": (255, 255, 0),
    "bicycle": (255, 200, 0),
    "traffic_cone": (0, 0, 255),
    "barrier": (200, 200, 200),
}

IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def box3d_to_corners(boxes_np):
    """Convert [N, 7+] boxes (x,y,z,w,l,h,yaw,...) to [N, 8, 3] corners."""
    x, y, z = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2]
    w, l, h = boxes_np[:, 3], boxes_np[:, 4], boxes_np[:, 5]
    yaw = boxes_np[:, 6]

    corners = np.array([
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [ 0.5,  0.5, -0.5],
    ])  # [8, 3]

    n = len(boxes_np)
    dims = np.stack([w, l, h], axis=-1)  # [N, 3]
    corners_3d = corners[None, :, :] * dims[:, None, :]  # [N, 8, 3]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.zeros((n, 3, 3))
    rot[:, 0, 0] = cos_yaw
    rot[:, 0, 1] = -sin_yaw
    rot[:, 1, 0] = sin_yaw
    rot[:, 1, 1] = cos_yaw
    rot[:, 2, 2] = 1.0

    corners_3d = np.einsum("nij,nmj->nmi", rot, corners_3d)

    center = np.stack([x, y, z], axis=-1)[:, None, :]
    corners_3d = corners_3d + center

    return corners_3d


def draw_box3d_on_img(img, bboxes3d, proj_mat, cls_indices, thickness=2):
    """Draw 3D bboxes on a single camera image with class colors."""
    if len(bboxes3d) == 0:
        return img

    corners_3d = box3d_to_corners(bboxes3d)
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
    )
    proj = copy.deepcopy(proj_mat).reshape(4, 4)
    if isinstance(proj, torch.Tensor):
        proj = proj.cpu().numpy()

    pts_2d = pts_4d @ proj.T
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    h, w = img.shape[:2]
    line_pairs = [
        (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2),
        (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7),
    ]
    bottom_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for i in range(num_bbox):
        corners = np.clip(pts_2d[i], -1e4, 1e5).astype(np.int32)
        visible = any(0 <= c[0] < w and 0 <= c[1] < h for c in corners)
        if not visible:
            continue

        cls_name = CLASS_NAMES[cls_indices[i]] if cls_indices[i] < len(CLASS_NAMES) else "unknown"
        color = CLASS_COLORS.get(cls_name, (0, 255, 0))

        for s, e in line_pairs:
            cv2.line(img, tuple(corners[s]), tuple(corners[e]), color, thickness, cv2.LINE_AA)
        for s, e in bottom_pairs:
            cv2.line(img, tuple(corners[s]), tuple(corners[e]), color, thickness + 1, cv2.LINE_AA)

        top_y = min(corners[:, 1])
        top_x = int(np.mean(corners[:, 0]))
        if 0 < top_x < w and 0 < top_y < h:
            cv2.putText(img, cls_name, (top_x - 20, max(top_y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return img


def make_frame(raw_imgs, bboxes3d, proj_mats, cls_indices, scores, frame_idx, elapsed_ms):
    """Create a visualization frame: 3x2 camera grid with info overlay."""
    h, w = raw_imgs.shape[1], raw_imgs.shape[2]  # 256, 704

    scale = 2
    sh, sw = h * scale, w * scale

    cam_order = [2, 0, 1, 4, 3, 5]  # FL, F, FR, BL, B, BR
    cam_names = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "BACK_LEFT", "BACK", "BACK_RIGHT"]

    rows = []
    for row in range(2):
        cols = []
        for col in range(3):
            idx = cam_order[row * 3 + col]
            img = raw_imgs[idx].clip(0, 255).astype(np.uint8).copy()

            if len(bboxes3d) > 0:
                img = draw_box3d_on_img(img, bboxes3d, proj_mats[idx], cls_indices)

            img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)

            cv2.putText(img, cam_names[row * 3 + col], (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cols.append(img)
        rows.append(np.concatenate(cols, axis=1))

    canvas = np.concatenate(rows, axis=0)

    # Info bar
    info_h = 50
    info_bar = np.zeros((info_h, canvas.shape[1], 3), dtype=np.uint8)
    num_det = len(bboxes3d)

    cv2.putText(
        info_bar,
        f"Frame {frame_idx:04d}  |  Detections: {num_det}  |  TT-NN {elapsed_ms:.0f}ms  |  Sparse4D N300 Mesh SPMD",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )

    if num_det > 0:
        cls_counts = {}
        for c in cls_indices:
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else "?"
            cls_counts[name] = cls_counts.get(name, 0) + 1
        x_offset = canvas.shape[1] // 2
        for name, cnt in cls_counts.items():
            color = CLASS_COLORS.get(name, (0, 255, 0))
            text = f"{name}: {cnt}"
            cv2.putText(info_bar, text, (x_offset, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            x_offset += len(text) * 12 + 20

    canvas = np.concatenate([canvas, info_bar], axis=0)
    return canvas


def decode_outputs(outputs, conf_threshold=0.3, mesh_device=None):
    """Decode TT-NN model outputs to boxes, classes, scores."""
    prediction = outputs["prediction"][-1]
    classification = outputs["classification"][-1]
    quality_list = outputs.get("quality", [])
    quality = quality_list[-1] if quality_list and quality_list[-1] is not None else None

    if prediction is None or classification is None:
        return np.array([]), np.array([]), np.array([])

    def _to_torch(t):
        if isinstance(t, torch.Tensor):
            return t.float()
        if mesh_device is not None:
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:1]
        return ttnn.to_torch(t).float()

    pred = _to_torch(prediction)[0]  # [num_anchor, 11]
    cls = _to_torch(classification)[0]  # [num_anchor, num_cls]
    cls_scores = cls.sigmoid()

    cls_scores_max, cls_ids = cls_scores.max(dim=-1)

    # Quality weighting
    if quality is not None:
        qt = _to_torch(quality)[0]
        centerness = qt[..., 0].sigmoid()
        cls_scores_max = cls_scores_max * centerness

    mask = cls_scores_max > conf_threshold
    if mask.sum() == 0:
        return np.array([]), np.array([]), np.array([])

    filtered_pred = pred[mask]
    filtered_cls = cls_ids[mask]
    filtered_scores = cls_scores_max[mask]

    # Decode: x,y,z,exp(w),exp(l),exp(h),yaw,vx,vy,vz
    yaw = torch.atan2(filtered_pred[:, 6], filtered_pred[:, 7])
    decoded = torch.cat([
        filtered_pred[:, :3],
        filtered_pred[:, 3:6].exp(),
        yaw.unsqueeze(-1),
        filtered_pred[:, 8:],
    ], dim=-1).numpy()

    return decoded, filtered_cls.numpy(), filtered_scores.numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Sparse4D TT-NN Inference Video Generator (Mesh SPMD + BF16)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["video", "images", "fps"], default="video",
                        help="output mode (default: video)")
    parser.add_argument("--frames", type=int, default=80, help="frames to process (default: 80)")
    parser.add_argument("--conf", type=float, default=0.3, help="confidence threshold (default: 0.3)")
    parser.add_argument("--outdir", type=str, default="inference_output", help="output dir for images mode")
    parser.add_argument("--output", type=str, default="inference_video_tt.mp4", help="output video path")
    parser.add_argument("--fps", type=int, default=4, help="video fps (default: 4)")
    parser.add_argument("--ckpt", type=str, default="ckpt/latest.pth", help="checkpoint path")
    parser.add_argument("--data-root", type=str, default="nuscenes/trainval", help="nuScenes data root")
    parser.add_argument("--anno-pkl", type=str, default="nuscenes_anno_pkls/nuscenes_infos_val.pkl")
    parser.add_argument("--dual-device", action="store_true", help="use mesh SPMD dual device")
    args = parser.parse_args()

    from loguru import logger as _logger
    _logger.remove()
    _logger.add(sys.stderr, level="WARNING")

    print("=" * 60)
    print("  Sparse4D TT-NN Inference Video (Mesh SPMD + BF16)")
    print("=" * 60)

    # 1. Load data
    print("\n[1/3] Loading nuScenes data...")
    loader = NuScenesValLoader(args.data_root, args.anno_pkl)

    # 2. Build model
    print("\n[2/3] Building TT-NN model...")
    num_devices = ttnn.get_num_devices()
    mesh_device = None

    if args.dual_device and num_devices >= 2:
        print(f"  {num_devices} chips detected, opening mesh (1x2)...")
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), l1_small_size=24576)
        device = mesh_device
        print(f"  Mesh device IDs: {mesh_device.get_device_ids()}")

        model = load_model(args.ckpt, mesh_device, mesh_device=mesh_device, backbone_batch_size=3, fp32_backbone=False)
        model.mesh_parallel_mode = True
        model._mesh_device = mesh_device
        model.serial_cams_per_batch = 0
        print("  Full mesh SPMD mode: backbone+FPN+Head all on mesh_device (bf16)")
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=24576)
        model = load_model(args.ckpt, device, backbone_batch_size=6)
        model.serial_cams_per_batch = 0
        print("  Single device mode")

    # 3. Run inference + generate output
    print(f"\n[3/3] Running inference ({args.frames} frames, mode={args.mode})...")

    if args.mode == "images":
        os.makedirs(args.outdir, exist_ok=True)

    writer = None
    times = []
    processed = 0

    for scene_idx, scene_samples in enumerate(loader.scenes):
        model.reset()

        for sample_idx, global_idx in enumerate(scene_samples):
            if processed >= args.frames:
                break

            images, metas, info = loader.get_sample(global_idx)

            t0 = time.time()
            outputs = model.forward(images, metas, bs=1)
            elapsed = time.time() - t0
            elapsed_ms = elapsed * 1000
            times.append(elapsed_ms)
            processed += 1

            # Periodic GC
            if processed % 50 == 0:
                gc.collect()

            # Decode predictions
            decoded_boxes, cls_indices, scores = decode_outputs(
                outputs, conf_threshold=args.conf, mesh_device=mesh_device
            )
            num_det = len(decoded_boxes)

            if args.mode == "fps":
                print(f"  Frame {processed:03d}: {elapsed_ms:6.1f} ms ({1000/elapsed_ms:5.1f} fps) {num_det:2d} det")
                continue

            # Denormalize images for visualization
            raw_imgs = images[0].permute(0, 2, 3, 1).numpy()  # [6, H, W, 3]
            raw_imgs = raw_imgs * IMG_STD + IMG_MEAN

            proj_mats = metas["projection_mat"][0].numpy()  # [6, 4, 4]

            frame = make_frame(
                raw_imgs, decoded_boxes, proj_mats,
                cls_indices, scores, processed - 1, elapsed_ms,
            )

            if args.mode == "images":
                out_path = os.path.join(args.outdir, f"frame_{processed-1:04d}.png")
                cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
                    if not writer.isOpened():
                        # Fallback: try x264 or mp4v if avc1 not available
                        for codec in ["x264", "H264", "mp4v"]:
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
                            if writer.isOpened():
                                print(f"  Video: {w}x{h}, {args.fps} fps, codec={codec} -> {args.output}")
                                break
                        if not writer.isOpened():
                            print(f"  WARNING: No H.264 encoder found, falling back to mp4v")
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
                    else:
                        print(f"  Video: {w}x{h}, {args.fps} fps, codec=h264 -> {args.output}")
                writer.write(frame_bgr)

            max_score = scores.max() if len(scores) > 0 else 0
            print(f"  Frame {processed-1:03d}: {num_det:2d} det (max: {max_score:.3f}) [{elapsed_ms:.0f}ms]")

        if processed >= args.frames:
            break

    # Summary
    if times:
        avg_ms = np.mean(times)
        std_ms = np.std(times)
        print(f"\n{'=' * 50}")
        print(f"  Inference: {avg_ms:.1f} +/- {std_ms:.1f} ms/frame")
        print(f"  FPS:       {1000/avg_ms:.1f}")
        print(f"  Frames:    {processed}")
        print(f"{'=' * 50}")

    if args.mode == "video" and writer is not None:
        writer.release()
        print(f"\nSaved: {args.output} ({processed} frames)")
    elif args.mode == "images":
        print(f"\nSaved: {processed} images to {args.outdir}/")

    # Cleanup
    if mesh_device is not None:
        ttnn.close_mesh_device(mesh_device)
    else:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
