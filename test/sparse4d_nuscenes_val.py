"""
nuScenes Validation with TT-NN Sparse4D (standalone, no mmcv)

Loads checkpoint directly via torch.load, builds TT-NN model,
runs inference on nuScenes val set, evaluates mAP/NDS.

Usage:
  python test/sparse4d_nuscenes_val.py
  python test/sparse4d_nuscenes_val.py --num-samples 50
  python test/sparse4d_nuscenes_val.py --ckpt ckpt/latest.pth --data-root nuscenes/trainval
"""

import argparse
import copy
import gc
import json
import os
import pickle
import sys
import tempfile
import time

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.expanduser("~/project/tt-metal")
)
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import ttnn
from PIL import Image
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion

from model.fpn import FPN, _ConvParams, _FPNParameters, _WeightBias
from model.sparse4d import IMG_MEAN, IMG_STD, SPATIAL_SHAPES, Sparse4DInference
from model.sparse4d_head import Sparse4DHead
from model.resnet_bottleneck import (
    TtResNetBottleneck,
    create_tt_resnet_bottleneck,
    infer_conv_shapes,
    preprocess_resnet50_parameters,
)

# ============================================================
# 1. Build TT-NN model from checkpoint state_dict
# ============================================================

# Must match the training config's class_names order (model output label indices)
# From: sparse4dv3_temporal_r50_1x8_bs6_256x704.py
CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

OPERATION_ORDER = [
    "deformable",
    "ffn",
    "norm",
    "refine",
    "temp_gnn",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
    "temp_gnn",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
    "temp_gnn",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
    "temp_gnn",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
    "temp_gnn",
    "gnn",
    "norm",
    "deformable",
    "ffn",
    "norm",
    "refine",
]


def _build_resnet_from_sd(sd, device, batch_size=6):
    """Build TT ResNet50 backbone from checkpoint state_dict."""
    import torchvision

    resnet = torchvision.models.resnet50(weights=None)
    resnet_sd = {
        k.replace("img_backbone.", ""): v
        for k, v in sd.items()
        if k.startswith("img_backbone.")
    }
    resnet.load_state_dict(resnet_sd, strict=False)
    resnet.eval()

    backbone, _ = create_tt_resnet_bottleneck(
        torch_model=resnet,
        device=device,
        batch_size=batch_size,
        input_height=256,
        input_width=704,
    )
    return backbone


def _build_fpn_from_sd(sd, device, batch_size=6, fp32=False):
    """Build TT FPN from checkpoint state_dict."""
    lateral_params = []
    fpn_params = []

    for i in range(4):
        w = sd[f"img_neck.lateral_convs.{i}.conv.weight"]
        b = sd[f"img_neck.lateral_convs.{i}.conv.bias"]
        lateral_params.append(
            {
                "weight": ttnn.from_torch(w),
                "bias": ttnn.from_torch(b.reshape(1, 1, 1, -1)),
            }
        )

        w = sd[f"img_neck.fpn_convs.{i}.conv.weight"]
        b = sd[f"img_neck.fpn_convs.{i}.conv.bias"]
        fpn_params.append(
            {
                "weight": ttnn.from_torch(w),
                "bias": ttnn.from_torch(b.reshape(1, 1, 1, -1)),
            }
        )

    parameters = _FPNParameters(lateral_params, fpn_params)

    act_dtype = ttnn.float32 if fp32 else ttnn.bfloat16
    weights_dtype = ttnn.float32 if fp32 else ttnn.bfloat16
    math_fidelity = ttnn.MathFidelity.HiFi4 if fp32 else ttnn.MathFidelity.LoFi

    fpn = FPN(
        device=device,
        parameters=parameters,
        batch_size=batch_size,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        model_config={
            "WEIGHTS_DTYPE": weights_dtype,
            "ACTIVATIONS_DTYPE": act_dtype,
            "MATH_FIDELITY": math_fidelity,
        },
        input_spatial_shapes=SPATIAL_SHAPES,
    )
    return fpn


def _extract_linear_relu_ln(sd, prefix, max_idx=None):
    """Extract Linear→ReLU→LN chain params from state_dict.

    Parses nn.Sequential with structure:
      idx 0: Linear (weight, bias)
      idx 1: ReLU
      idx 2: LayerNorm (weight, bias)
      idx 3: Linear ...

    Args:
        max_idx: Stop before this index (exclusive). Used to exclude
                 final linear/scale layers from the chain.
    """
    params = []
    idx = 0
    while True:
        if max_idx is not None and idx >= max_idx:
            break
        w_key = f"{prefix}.{idx}.weight"
        if w_key not in sd:
            break
        w = sd[w_key]
        b = sd[f"{prefix}.{idx}.bias"]

        # Check if this is a Linear (2D weight) or LayerNorm (1D weight)
        if w.dim() == 2:
            entry = {"weight": w.t(), "bias": b}
            # Check if idx+2 is LayerNorm (1D weight)
            ln_key = f"{prefix}.{idx + 2}.weight"
            if ln_key in sd and sd[ln_key].dim() == 1:
                entry["ln_weight"] = sd[ln_key]
                entry["ln_bias"] = sd[f"{prefix}.{idx + 2}.bias"]
                idx += 3  # Skip Linear, ReLU, LayerNorm
            else:
                idx += 2  # Skip Linear, ReLU
            params.append(entry)
        else:
            # 1D weight = LayerNorm, skip (already handled above)
            idx += 1
    return params


def _build_mha_params(sd, prefix):
    """Extract MHA params from state_dict."""
    iw = sd[f"{prefix}.attn.in_proj_weight"]
    ib = sd[f"{prefix}.attn.in_proj_bias"]
    wq, wk, wv = iw.chunk(3, dim=0)
    bq, bk, bv = ib.chunk(3, dim=0)
    return {
        "w_q": wq.t(),
        "w_k": wk.t(),
        "w_v": wv.t(),
        "b_q": bq,
        "b_k": bk,
        "b_v": bv,
        "w_out": sd[f"{prefix}.attn.out_proj.weight"].t(),
        "b_out": sd[f"{prefix}.attn.out_proj.bias"],
    }


def _build_norm_params(sd, prefix):
    """Extract LayerNorm params from state_dict."""
    return {
        "weight": sd[f"{prefix}.weight"],
        "bias": sd[f"{prefix}.bias"],
    }


def _build_ffn_params(sd, prefix):
    """Extract AsymmetricFFN params from state_dict."""
    return {
        "pre_norm_weight": sd[f"{prefix}.pre_norm.weight"],
        "pre_norm_bias": sd[f"{prefix}.pre_norm.bias"],
        "fc1_weight": sd[f"{prefix}.layers.0.0.weight"].t(),
        "fc1_bias": sd[f"{prefix}.layers.0.0.bias"],
        "fc2_weight": sd[f"{prefix}.layers.1.weight"].t(),
        "fc2_bias": sd[f"{prefix}.layers.1.bias"],
        "identity_fc_weight": sd[f"{prefix}.identity_fc.weight"].t(),
        "identity_fc_bias": sd[f"{prefix}.identity_fc.bias"],
    }


def _build_dfa_params(sd, prefix):
    """Extract DeformableFeatureAggregation params from state_dict."""
    return {
        "kps_fix_scale": sd[f"{prefix}.kps_generator.fix_scale"],
        "kps_learnable_fc_weight": sd[f"{prefix}.kps_generator.learnable_fc.weight"],
        "kps_learnable_fc_bias": sd[f"{prefix}.kps_generator.learnable_fc.bias"],
        "cam_linear1_weight": sd[f"{prefix}.camera_encoder.0.weight"],
        "cam_linear1_bias": sd[f"{prefix}.camera_encoder.0.bias"],
        "cam_ln1_weight": sd[f"{prefix}.camera_encoder.2.weight"],
        "cam_ln1_bias": sd[f"{prefix}.camera_encoder.2.bias"],
        "cam_linear2_weight": sd[f"{prefix}.camera_encoder.3.weight"],
        "cam_linear2_bias": sd[f"{prefix}.camera_encoder.3.bias"],
        "cam_ln2_weight": sd[f"{prefix}.camera_encoder.5.weight"],
        "cam_ln2_bias": sd[f"{prefix}.camera_encoder.5.bias"],
        "weights_fc_weight": sd[f"{prefix}.weights_fc.weight"],
        "weights_fc_bias": sd[f"{prefix}.weights_fc.bias"],
        "output_proj_weight": sd[f"{prefix}.output_proj.weight"],
        "output_proj_bias": sd[f"{prefix}.output_proj.bias"],
    }


def _build_refine_params(sd, prefix):
    """Extract SparseBox3DRefinementModule params from state_dict."""
    # Find final linear idx and scale idx in refine layers
    scale_key = None
    final_idx = None
    for k in sd:
        if k.startswith(f"{prefix}.layers.") and k.endswith(".scale"):
            scale_idx = int(k.split(".")[len(prefix.split(".")) + 1])
            scale_key = k
            final_idx = scale_idx - 1
            break

    # Refine layers: Linear→ReLU→LN chains (exclude final linear and scale)
    refine_layers = _extract_linear_relu_ln(sd, f"{prefix}.layers", max_idx=final_idx)

    params = {
        "refine_layers": refine_layers,
        "refine_final_weight": sd[f"{prefix}.layers.{final_idx}.weight"].t(),
        "refine_final_bias": sd[f"{prefix}.layers.{final_idx}.bias"],
        "refine_scale": sd[scale_key],
    }

    # Classification layers: find final linear idx
    last_cls_idx = _find_last_linear_idx(sd, f"{prefix}.cls_layers")
    params["cls_layers"] = _extract_linear_relu_ln(
        sd, f"{prefix}.cls_layers", max_idx=last_cls_idx
    )
    params["cls_final_weight"] = sd[f"{prefix}.cls_layers.{last_cls_idx}.weight"].t()
    params["cls_final_bias"] = sd[f"{prefix}.cls_layers.{last_cls_idx}.bias"]

    # Quality layers
    if f"{prefix}.quality_layers.0.weight" in sd:
        last_qt_idx = _find_last_linear_idx(sd, f"{prefix}.quality_layers")
        params["quality_layers"] = _extract_linear_relu_ln(
            sd, f"{prefix}.quality_layers", max_idx=last_qt_idx
        )
        params["quality_final_weight"] = sd[
            f"{prefix}.quality_layers.{last_qt_idx}.weight"
        ].t()
        params["quality_final_bias"] = sd[f"{prefix}.quality_layers.{last_qt_idx}.bias"]

    return params


def _find_last_linear_idx(sd, prefix):
    """Find the index of the last Linear layer (2D weight) under prefix."""
    return max(
        int(k.split(".")[len(prefix.split("."))])
        for k in sd
        if k.startswith(f"{prefix}.") and k.endswith(".weight") and sd[k].dim() == 2
    )


def _build_encoder_params(sd, prefix):
    """Extract SparseBox3DEncoder params from state_dict."""
    params = {}
    for fc_name in ["pos_fc", "size_fc", "yaw_fc", "vel_fc"]:
        fc_prefix = f"{prefix}.{fc_name}"
        if f"{fc_prefix}.0.weight" not in sd:
            continue
        params[fc_name] = _extract_linear_relu_ln(sd, fc_prefix)
    return params


def _build_head_from_sd(sd, device, mesh_device=None):
    """Build TT Sparse4DHead from checkpoint state_dict."""
    layer_params = []
    for i, op in enumerate(OPERATION_ORDER):
        prefix = f"head.layers.{i}"
        if op in ("gnn", "temp_gnn"):
            layer_params.append(_build_mha_params(sd, prefix))
        elif op == "norm":
            layer_params.append(_build_norm_params(sd, prefix))
        elif op == "deformable":
            layer_params.append(_build_dfa_params(sd, prefix))
        elif op == "ffn":
            layer_params.append(_build_ffn_params(sd, prefix))
        elif op == "refine":
            layer_params.append(_build_refine_params(sd, prefix))

    params = {
        "layers": layer_params,
        "anchor_encoder": _build_encoder_params(sd, "head.anchor_encoder"),
        "instance_bank": {
            "anchor_data": sd["head.instance_bank.anchor"],
            "instance_feature_data": sd["head.instance_bank.instance_feature"],
        },
    }

    if "head.fc_before.weight" in sd:
        params["fc_before_weight"] = sd["head.fc_before.weight"].t()
        params["fc_after_weight"] = sd["head.fc_after.weight"].t()

    head = Sparse4DHead(
        device=device,
        parameters=params,
        operation_order=OPERATION_ORDER,
        embed_dims=256,
        num_decoder=6,
        num_single_frame_decoder=1,
        num_anchor=900,
        num_temp_instances=600,
        num_classes=10,
        num_groups=8,
        spatial_shapes=SPATIAL_SHAPES,
        mesh_device=mesh_device,
    )
    return head


def load_model(ckpt_path, device, mesh_device=None, backbone_batch_size=None, fp32_backbone=False):
    """Load full TT-NN Sparse4D model from checkpoint.

    Args:
        ckpt_path: path to checkpoint
        device: single TT device
        mesh_device: deprecated, ignored
        backbone_batch_size: override batch size for backbone/FPN (default: 6)
        fp32_backbone: use float32 for backbone/FPN activations
    """
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    print(f"  State dict: {len(sd)} parameters")

    batch_size = backbone_batch_size or 6
    print(f"  Building ResNet50 backbone (batch={batch_size}, fp32={fp32_backbone})...")
    backbone = _build_resnet_from_sd(sd, device, batch_size=batch_size)

    print(f"  Building FPN (batch={batch_size}, fp32={fp32_backbone})...")
    fpn = _build_fpn_from_sd(sd, device, batch_size=batch_size, fp32=fp32_backbone)

    print("  Building Sparse4DHead...")
    head = _build_head_from_sd(sd, device, mesh_device=mesh_device)

    model = Sparse4DInference(device, backbone, fpn, head)
    print("  Model build complete.")
    return model


# ============================================================
# 2. Standalone nuScenes Data Loader
# ============================================================


def _compute_lidar2img(cam_info):
    """Compute lidar-to-image projection matrix for a camera."""
    lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
    lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    intrinsic = cam_info["cam_intrinsic"].copy()
    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img = viewpad @ lidar2cam_rt.T
    return lidar2img


def _compute_lidar2global(info):
    """Compute lidar-to-global transformation matrix."""
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
    ego2global[:3, 3] = np.array(info["ego2global_translation"])

    return ego2global @ lidar2ego


def _resize_crop_image(img, resize, crop):
    """Resize and crop image, return transformed image and matrix."""
    w, h = img.size
    img = img.resize((int(w * resize), int(h * resize)), Image.BILINEAR)
    img = img.crop(crop)

    # Build transform matrix
    transform = np.eye(4)
    transform[0, 0] = resize
    transform[1, 1] = resize
    transform[0, 2] = -crop[0] * resize  # Hmm, actually...
    transform[1, 2] = -crop[1] * resize

    # Actually the correct transform:
    # 1. Scale: pixel coords scaled by resize
    # 2. Translate: subtract crop offset
    transform_3x3 = np.eye(3)
    transform_3x3[:2, :2] *= resize
    transform_3x3[0, 2] = -crop[0]
    transform_3x3[1, 2] = -crop[1]

    mat_4x4 = np.eye(4)
    mat_4x4[:3, :3] = transform_3x3
    return img, mat_4x4


class NuScenesValLoader:
    """Standalone nuScenes validation data loader."""

    IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(self, data_root, anno_pkl, target_size=(256, 704)):
        """
        Args:
            data_root: Path to nuScenes data (contains samples/, sweeps/, v1.0-trainval/)
            anno_pkl: Path to nuscenes_infos_val.pkl
            target_size: (H, W) target image size
        """
        self.data_root = data_root
        self.target_h, self.target_w = target_size

        with open(anno_pkl, "rb") as f:
            data = pickle.load(f)
        self.infos = sorted(data["infos"], key=lambda e: e["timestamp"])
        self.camera_types = None  # Set from first info

        # Group by scene for temporal ordering
        self._build_scene_groups()

        print(f"  Loaded {len(self.infos)} val samples, {len(self.scenes)} scenes")

    def _build_scene_groups(self):
        """Group samples by scene for temporal processing."""
        self.scenes = []
        current_scene = []
        prev_timestamp = None

        for i, info in enumerate(self.infos):
            ts = info["timestamp"]
            # New scene if time gap > 3 seconds
            if prev_timestamp is not None and abs(ts - prev_timestamp) > 3e6:
                if current_scene:
                    self.scenes.append(current_scene)
                current_scene = []
            current_scene.append(i)
            prev_timestamp = ts

        if current_scene:
            self.scenes.append(current_scene)

    def __len__(self):
        return len(self.infos)

    def _resolve_path(self, data_path):
        """Resolve image path from annotation to actual file."""
        # Try the path as-is first
        if os.path.exists(data_path):
            return data_path

        # Try relative to data_root
        # Strip common prefixes like 'data/nuscenes/' or './data/nuscenes/'
        for prefix in ["data/nuscenes/", "./data/nuscenes/", "nuscenes/"]:
            if data_path.startswith(prefix):
                candidate = os.path.join(self.data_root, data_path[len(prefix) :])
                if os.path.exists(candidate):
                    return candidate

        # Try just the filename parts (samples/CAM_X/file.jpg)
        parts = data_path.split("/")
        for start_idx in range(len(parts)):
            candidate = os.path.join(self.data_root, "/".join(parts[start_idx:]))
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            f"Cannot find image: {data_path} (data_root={self.data_root})"
        )

    def get_sample(self, idx):
        """Load and preprocess a single sample.

        Returns:
            images: [1, 6, 3, H, W] normalized tensor
            metas: dict with projection_mat, image_wh, timestamp, img_metas
        """
        info = self.infos[idx]

        if self.camera_types is None:
            self.camera_types = list(info["cams"].keys())

        # Original image size (nuScenes: 900x1600, H x W)
        orig_h, orig_w = 900, 1600

        # Test-time resize: max(target_h/orig_h, target_w/orig_w)
        resize = max(self.target_h / orig_h, self.target_w / orig_w)
        new_w, new_h = int(orig_w * resize), int(orig_h * resize)

        # Center crop
        crop_h = new_h - self.target_h
        crop_w = max(0, new_w - self.target_w) // 2
        crop = (crop_w, crop_h, crop_w + self.target_w, crop_h + self.target_h)

        imgs = []
        lidar2img_list = []

        for cam_type in self.camera_types:
            cam_info = info["cams"][cam_type]

            # Load image
            img_path = self._resolve_path(cam_info["data_path"])
            img = Image.open(img_path).convert("RGB")

            # Resize and crop
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img = img.crop(crop)
            img_np = np.array(img, dtype=np.float32)  # [H, W, 3]

            # Normalize
            img_np = (img_np - self.IMG_MEAN) / self.IMG_STD

            # HWC → CHW
            imgs.append(img_np.transpose(2, 0, 1))

            # Compute lidar2img with resize/crop applied
            lidar2img = _compute_lidar2img(cam_info)
            # Apply resize + crop transform
            aug_mat = np.eye(4)
            aug_mat[0, 0] = resize
            aug_mat[1, 1] = resize
            aug_mat[0, 2] = -crop[0]
            aug_mat[1, 2] = -crop[1]
            lidar2img = aug_mat @ lidar2img
            lidar2img_list.append(lidar2img)

        # Stack
        images = (
            torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(0).float()
        )  # [1, 6, 3, H, W]

        projection_mat = torch.from_numpy(
            np.stack(lidar2img_list, axis=0).astype(np.float32)
        ).unsqueeze(0)  # [1, 6, 4, 4]

        image_wh = torch.tensor(
            [[self.target_w, self.target_h]] * len(self.camera_types),
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 6, 2]

        lidar2global = _compute_lidar2global(info)

        timestamp = torch.tensor([info["timestamp"] / 1e6], dtype=torch.float64)

        metas = {
            "projection_mat": projection_mat,
            "image_wh": image_wh,
            "timestamp": timestamp,
            "img_metas": [
                {
                    "T_global": lidar2global.astype(np.float32),
                    "T_global_inv": np.linalg.inv(lidar2global).astype(np.float32),
                    "timestamp": info["timestamp"] / 1e6,
                }
            ],
        }

        return images, metas, info


# ============================================================
# 3. Post-processing: TT-NN output → nuScenes format
# ============================================================


def _yaw_from_sincos(sin_yaw, cos_yaw):
    """Convert sin/cos yaw to angle."""
    return np.arctan2(sin_yaw, cos_yaw)


def _yaw_to_quaternion(yaw):
    """Convert yaw angle to quaternion [w, x, y, z]."""
    return Quaternion(axis=[0, 0, 1], radians=float(yaw))


def postprocess_to_nuscenes(
    outputs,
    info,
    eval_config,
    max_dets=300,
    mesh_device=None,
):
    """Convert TT-NN predictions to nuScenes detection format.

    Uses PyTorch-identical pipeline:
    1. squeeze_cls: max per anchor → topk(300)
    2. quality centerness weighting
    3. decode boxes (exp for w,l,h, atan2 for yaw)
    4. NuScenesBox: lidar → ego (+ range filter) → global
    5. velocity-based attributes

    Args:
        outputs: dict from Sparse4DInference.forward()
        info: sample info dict from pkl (has lidar2ego, ego2global transforms)
        eval_config: nuScenes eval config for class range filtering
        max_dets: maximum detections per sample (topk)

    Returns:
        list of dicts in nuScenes submission format
    """
    prediction = outputs["prediction"][-1]
    classification = outputs["classification"][-1]
    quality_list = outputs.get("quality", [])
    quality = quality_list[-1] if quality_list and quality_list[-1] is not None else None

    if prediction is None or classification is None:
        return []

    # To host
    def _tt_to_torch(t, mesh_dev=None):
        if isinstance(t, torch.Tensor):
            return t.float()
        if mesh_dev is not None:
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_dev, dim=0)).float()[:1]
        return ttnn.to_torch(t).float()

    pred = _tt_to_torch(prediction, mesh_device)
    cls = _tt_to_torch(classification, mesh_device)

    if quality is not None:
        qt = _tt_to_torch(quality, mesh_device)
        qt = qt[0]  # [num_anchor, num_quality]
    else:
        qt = None

    pred = pred[0]  # [num_anchor, 11]
    cls = cls[0]  # [num_anchor, num_cls]

    num_cls = cls.shape[-1]
    cls_scores = cls.sigmoid()  # [num_anchor, num_cls]

    # PyTorch original style (squeeze_cls): max per anchor, then topk
    cls_scores_max, cls_ids = cls_scores.max(dim=-1)  # [num_anchor], [num_anchor]
    cls_scores_max = cls_scores_max.unsqueeze(-1)  # [num_anchor, 1]

    scores_flat, indices = cls_scores_max.flatten().topk(max_dets)
    labels = cls_ids[indices]
    anchor_indices = indices

    # Apply quality centerness weighting
    if qt is not None:
        centerness = qt[..., 0]  # [num_anchor] — CNS index 0
        centerness_selected = centerness[anchor_indices]
        scores_flat = scores_flat * centerness_selected.sigmoid()
        # Re-sort after weighting
        scores_flat, sort_idx = scores_flat.sort(descending=True)
        labels = labels[sort_idx]
        anchor_indices = anchor_indices[sort_idx]

    scores_np = scores_flat.numpy()
    labels_np = labels.numpy()

    # Decode boxes (same as SparseBox3DDecoder.decode_box)
    selected_pred = pred[anchor_indices]
    yaw = torch.atan2(selected_pred[:, 6], selected_pred[:, 7])  # SIN_YAW, COS_YAW
    decoded_boxes = torch.cat([
        selected_pred[:, :3],           # x, y, z
        selected_pred[:, 3:6].exp(),    # w, l, h (in meters)
        yaw.unsqueeze(-1),              # yaw
        selected_pred[:, 8:],           # vx, vy, vz
    ], dim=-1).numpy()

    if len(scores_np) == 0:
        return []

    # Convert to NuScenesBox and transform (identical to PyTorch pipeline)
    cls_range_map = eval_config.class_range
    detections = []

    for i in range(len(scores_np)):
        box_center = decoded_boxes[i, :3]
        # Swap W↔L for nuScenes convention (PyTorch: box_dims[:, [1, 0, 2]])
        box_dims = decoded_boxes[i, [4, 3, 5]]  # [L, W, H]
        box_yaw = float(decoded_boxes[i, 6])
        vx, vy = float(decoded_boxes[i, 7]), float(decoded_boxes[i, 8])

        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw)
        box = NuScenesBox(
            box_center, box_dims, quat,
            label=int(labels_np[i]),
            score=float(scores_np[i]),
            velocity=(vx, vy, 0.0),
        )

        # Lidar → Ego
        box.rotate(Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))

        # Filter by class range (in ego frame) — same as PyTorch
        cls_name = CLASS_NAMES[box.label] if box.label < len(CLASS_NAMES) else "unknown"
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map.get(cls_name, 50)
        if radius > det_range:
            continue

        # Ego → Global
        box.rotate(Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))

        # Velocity-based attribute
        attr = _get_attribute(cls_name, box.velocity[:2])

        detections.append(
            {
                "translation": box.center.tolist(),
                "size": box.wlh.tolist(),
                "rotation": box.orientation.elements.tolist(),
                "velocity": box.velocity[:2].tolist(),
                "detection_name": cls_name,
                "detection_score": float(box.score),
                "attribute_name": attr,
            }
        )

    return detections


def _get_attribute(cls_name, velocity):
    """Determine attribute based on velocity magnitude (matches PyTorch original).

    Args:
        cls_name: detection class name
        velocity: [vx, vy] in global frame
    """
    DEFAULT_ATTR = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }

    speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    if speed > 0.2:
        if cls_name in ["car", "construction_vehicle", "bus", "truck", "trailer"]:
            return "vehicle.moving"
        elif cls_name in ["bicycle", "motorcycle"]:
            return "cycle.with_rider"
        else:
            return DEFAULT_ATTR.get(cls_name, "")
    else:
        if cls_name == "pedestrian":
            return "pedestrian.standing"
        elif cls_name == "bus":
            return "vehicle.stopped"
        else:
            return DEFAULT_ATTR.get(cls_name, "")


# ============================================================
# 4. nuScenes Evaluation
# ============================================================


def evaluate_partial(results_dict, data_root, version="v1.0-trainval", dist_thresh=2.0):
    """Evaluate only on processed samples (no filling with empty).

    Simple matching: for each GT box in processed samples, check if any
    detection is within dist_thresh meters (center distance, BEV).
    """
    from nuscenes import NuScenes

    print(f"  Loading NuScenes for partial eval...")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)

    # Use module-level CLASS_NAMES (matches MMDet ordering)
    # car, truck, trailer, bus, construction_vehicle,
    # bicycle, motorcycle, pedestrian, traffic_cone, barrier

    total_gt = {c: 0 for c in CLASS_NAMES}
    total_tp = {c: 0 for c in CLASS_NAMES}
    total_fp = {c: 0 for c in CLASS_NAMES}
    total_dets = 0

    for token, dets in results_dict.items():
        try:
            sample = nusc.get("sample", token)
        except Exception:
            continue

        # Collect GT boxes for this sample
        gt_boxes = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]
            det_name = None
            for dn in CLASS_NAMES:
                if dn in cat:
                    det_name = dn
                    break
            if det_name is None:
                continue
            pos = np.array(ann["translation"])
            gt_boxes.append({"name": det_name, "pos": pos, "matched": False})
            total_gt[det_name] += 1

        # Match detections to GT
        for d in dets:
            total_dets += 1
            d_pos = np.array(d["translation"])
            d_name = d["detection_name"]

            best_dist = float("inf")
            best_idx = -1
            for gi, gt in enumerate(gt_boxes):
                if gt["matched"] or gt["name"] != d_name:
                    continue
                dist = np.linalg.norm(d_pos[:2] - gt["pos"][:2])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = gi

            if best_dist < dist_thresh and best_idx >= 0:
                gt_boxes[best_idx]["matched"] = True
                total_tp[d_name] += 1
            else:
                total_fp[d_name] += 1

    # Print results
    print(f"\n  === Partial Evaluation ({len(results_dict)} samples, dist<{dist_thresh}m) ===")
    print(f"  {'Class':<25s} {'GT':>5s} {'TP':>5s} {'FP':>5s} {'Prec':>7s} {'Recall':>7s}")
    print(f"  {'-'*55}")

    total_gt_sum = 0
    total_tp_sum = 0
    total_fp_sum = 0
    for c in CLASS_NAMES:
        gt = total_gt[c]
        tp = total_tp[c]
        fp = total_fp[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / gt if gt > 0 else 0
        total_gt_sum += gt
        total_tp_sum += tp
        total_fp_sum += fp
        print(f"  {c:<25s} {gt:>5d} {tp:>5d} {fp:>5d} {prec:>7.3f} {rec:>7.3f}")

    overall_prec = total_tp_sum / (total_tp_sum + total_fp_sum) if (total_tp_sum + total_fp_sum) > 0 else 0
    overall_rec = total_tp_sum / total_gt_sum if total_gt_sum > 0 else 0
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<25s} {total_gt_sum:>5d} {total_tp_sum:>5d} {total_fp_sum:>5d} {overall_prec:>7.3f} {overall_rec:>7.3f}")
    print(f"  Total detections: {total_dets}")


def evaluate_nuscenes(results_dict, data_root, version="v1.0-trainval", subset_tokens=None):
    """Run nuScenes detection evaluation.

    Args:
        results_dict: {sample_token: [detection_dicts]}
        data_root: path to nuScenes data root
        version: nuScenes version
        subset_tokens: if provided, filter both pred and GT to only these tokens
                       (enables meaningful mAP on partial runs)

    Returns:
        metrics dict
    """
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    from nuscenes import NuScenes

    # Write results to temp JSON
    submission = {
        "results": results_dict,
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
    }

    tmpdir = tempfile.mkdtemp()
    result_path = os.path.join(tmpdir, "results_nusc.json")
    with open(result_path, "w") as f:
        json.dump(submission, f)

    print(f"  Results saved to {result_path}")
    print(f"  Loading NuScenes database from {data_root}...")

    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    eval_config = config_factory("detection_cvpr_2019")

    nusc_eval = DetectionEval(
        nusc,
        config=eval_config,
        result_path=result_path,
        eval_set="val",
        output_dir=tmpdir,
        verbose=True,
    )

    # Filter to subset if specified (enables partial evaluation with real mAP)
    if subset_tokens is not None:
        tokens_set = set(subset_tokens)
        nusc_eval.pred_boxes.boxes = {
            k: v for k, v in nusc_eval.pred_boxes.boxes.items() if k in tokens_set
        }
        nusc_eval.gt_boxes.boxes = {
            k: v for k, v in nusc_eval.gt_boxes.boxes.items() if k in tokens_set
        }
        nusc_eval.sample_tokens = list(tokens_set)
        print(f"  Filtered to {len(tokens_set)} subset tokens for partial eval")

    metrics_summary = nusc_eval.main(render_curves=False)

    # Print summary
    print("\n  === nuScenes Detection Results ===")
    try:
        import json as _json

        metrics_path = os.path.join(tmpdir, "metrics_summary.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as mf:
                ms = _json.load(mf)
            print(f"  NDS:  {ms.get('nd_score', 'N/A')}")
            print(f"  mAP:  {ms.get('mean_ap', 'N/A')}")
            for k in ["tp_errors"]:
                if k in ms:
                    for ek, ev in ms[k].items():
                        print(f"  {ek}: {ev}")
    except Exception as e:
        print(f"  (Could not parse metrics: {e})")

    return metrics_summary


# ============================================================
# 5. Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="nuScenes TT-NN Validation")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt/latest.pth",
        help="Checkpoint path",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="nuscenes/trainval",
        help="nuScenes data root (with samples/, v1.0-trainval/)",
    )
    parser.add_argument(
        "--anno-pkl",
        type=str,
        default="nuscenes_anno_pkls/nuscenes_infos_val.pkl",
        help="Annotation pickle path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples (None=all)",
    )
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip nuScenes evaluation (just run inference)",
    )
    parser.add_argument(
        "--single-device",
        action="store_true",
        help="(deprecated, now always single device) Kept for backward compatibility",
    )
    parser.add_argument(
        "--save-raw",
        type=str,
        default=None,
        help="Save raw model outputs (prediction, classification, quality) to .pt file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging (verbose model ops)",
    )
    parser.add_argument(
        "--dual-device",
        action="store_true",
        help="Use 2 devices (batch=3 each) for backbone+FPN with HiFi4+fp32_acc precision",
    )
    args = parser.parse_args()

    # Set log level: DEBUG only when --debug is passed
    from loguru import logger
    import sys
    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="WARNING")

    print("=" * 70)
    print("  nuScenes Validation with TT-NN Sparse4D")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading nuScenes data...")
    loader = NuScenesValLoader(args.data_root, args.anno_pkl)

    total_samples = args.num_samples or len(loader)
    total_samples = min(total_samples, len(loader))

    # 2. Open TT device & build model
    print("\n[2/4] Building TT-NN model...")

    num_devices = ttnn.get_num_devices()
    mesh_device = None
    submeshes = None

    if args.dual_device and num_devices >= 2:
        # Full mesh SPMD mode
        print(f"  {num_devices} chips detected, opening mesh (1x2)...")
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), l1_small_size=24576)
        device = mesh_device  # everything runs on mesh
        print(f"  Mesh device IDs: {mesh_device.get_device_ids()}")
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=24576)
        if num_devices >= 2:
            print(f"  {num_devices} chips detected, using single device")
        else:
            print("  Single device mode")

    try:
        if args.dual_device and mesh_device is not None:
            # Full mesh SPMD: backbone+FPN+Head all on mesh_device (bf16)
            # backbone batch=3 per device (SPMD), Head replicated on mesh
            model = load_model(args.ckpt, mesh_device, mesh_device=mesh_device, backbone_batch_size=3, fp32_backbone=False)

            model.mesh_parallel_mode = True
            model._mesh_device = mesh_device
            model.serial_cams_per_batch = 0

            print("  Full mesh SPMD mode: backbone+FPN+Head all on mesh_device (bf16)")
        elif args.dual_device:
            # Fallback: serial batch=3 x 2 on single device
            model = load_model(args.ckpt, device, mesh_device=None, backbone_batch_size=3, fp32_backbone=False)
            model.serial_cams_per_batch = 3
            print("  HiFi4+fp32_acc mode: serial batch=3 x 2 runs (single device)")
        else:
            # Single device mode: batch=6, all on-device
            model = load_model(args.ckpt, device, mesh_device=None, backbone_batch_size=6, fp32_backbone=False)
            model.serial_cams_per_batch = 0
            print("  Direct batch mode: 6 cameras, no serial (all on-device)")

        # 3. Run inference
        from nuscenes.eval.detection.config import config_factory
        eval_config = config_factory("detection_cvpr_2019")

        print(f"\n[3/4] Running inference on {total_samples} samples...")
        all_results = {}
        raw_outputs = []  # For --save-raw
        total_time = 0
        processed = 0

        # Process scene by scene for temporal coherence
        for scene_idx, scene_samples in enumerate(loader.scenes):
            model.reset()  # Reset temporal cache at scene boundary

            for sample_idx_in_scene, global_idx in enumerate(scene_samples):
                if processed >= total_samples:
                    break

                images, metas, info = loader.get_sample(global_idx)
                token = info["token"]

                t0 = time.time()
                outputs = model.forward(images, metas, bs=1)
                elapsed = time.time() - t0
                total_time += elapsed
                processed += 1

                # Save raw outputs if requested
                if args.save_raw:
                    raw_pred = outputs["prediction"][-1]
                    raw_cls = outputs["classification"][-1]
                    raw_qt_list = outputs.get("quality", [])
                    raw_qt = raw_qt_list[-1] if raw_qt_list and raw_qt_list[-1] is not None else None

                    def _to_cpu(t):
                        if t is None:
                            return None
                        if isinstance(t, torch.Tensor):
                            return t.cpu().float()
                        if mesh_device is not None:
                            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).cpu().float()[:1]
                        return ttnn.to_torch(t).cpu().float()

                    entry = {
                        "token": token,
                        "prediction": _to_cpu(raw_pred)[0],  # [900, 11]
                        "classification": _to_cpu(raw_cls)[0],  # [900, 10]
                    }
                    qt_val = _to_cpu(raw_qt)
                    if qt_val is not None:
                        entry["quality"] = qt_val[0]  # [900, 2]
                    raw_outputs.append(entry)

                # Post-process (PyTorch-identical: NuScenesBox + range filter)
                dets = postprocess_to_nuscenes(
                    outputs,
                    info,
                    eval_config,
                    mesh_device=mesh_device,
                )

                # Add sample_token to each detection
                for d in dets:
                    d["sample_token"] = token
                all_results[token] = dets

                # Periodic GC to prevent host memory leak from to_torch/from_torch
                if processed % 100 == 0:
                    gc.collect()

                if processed % 50 == 0 or processed == total_samples:
                    avg_time = total_time / processed
                    print(
                        f"  [{processed}/{total_samples}] "
                        f"{elapsed:.2f}s/sample, "
                        f"avg {avg_time:.2f}s, "
                        f"{len(dets)} dets"
                    )

            if processed >= total_samples:
                break

        avg_time = total_time / max(1, processed)
        print(
            f"\n  Inference complete: {processed} samples, avg {avg_time:.2f}s/sample"
        )

        # Save raw outputs if requested
        if args.save_raw and raw_outputs:
            torch.save(raw_outputs, args.save_raw)
            print(f"  Raw outputs saved to {args.save_raw} ({len(raw_outputs)} samples)")

        # 4. Evaluate
        if not args.skip_eval:
            if processed < len(loader):
                print(
                    f"\n[4/4] Running nuScenes evaluation "
                    f"({processed}/{len(loader)} samples)..."
                )
                # Partial evaluation: simple matching (TP/FP/Recall)
                evaluate_partial(all_results, args.data_root)

                # Partial nuScenes mAP: filter GT to only processed tokens
                processed_tokens = list(all_results.keys())
                print(f"\n  Running nuScenes mAP on {len(processed_tokens)} processed samples...")
                full_results = dict(all_results)
                for info in loader.infos:
                    if info["token"] not in full_results:
                        full_results[info["token"]] = []
                evaluate_nuscenes(full_results, args.data_root, subset_tokens=processed_tokens)
            else:
                print(f"\n[4/4] Running nuScenes evaluation...")
                evaluate_nuscenes(all_results, args.data_root)
        else:
            print(f"\n[4/4] Evaluation skipped (--skip-eval)")

            # Save results for later evaluation
            result_path = "tt_val_results.json"
            submission = {
                "results": all_results,
                "meta": {
                    "use_camera": True,
                    "use_lidar": False,
                    "use_radar": False,
                    "use_map": False,
                    "use_external": False,
                },
            }
            with open(result_path, "w") as f:
                json.dump(submission, f)
            print(f"  Results saved to {result_path}")

        print(f"\n{'=' * 70}")
        print(f"  Done. {processed} samples processed.")
        print(f"{'=' * 70}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\n  FAILED: {str(e)[:500]}")
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
        else:
            ttnn.close_device(device)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set
    main()
