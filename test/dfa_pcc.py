"""
DeformableFeatureAggregation PCC Test: PyTorch vs TT-NN (Device-Only)

Compares the TT-NN DeformableFeatureAggregation (device-only, all ttnn ops)
against the PyTorch fallback path (F.grid_sample) for numerical accuracy.

Metrics:
  - PCC: Pearson Correlation (pattern similarity)
  - MaxDiff: worst-case absolute error
  - MeanDiff: average absolute error
  - RelErr: mean relative error (|a-b| / max(|a|, |b|))
  - Allclose: torch.allclose with configurable atol/rtol
  - Sample values: side-by-side comparison of actual tensor values

Tests:
  1. grid_sample only: per-level feature sampling
  2. End-to-end: full DFA forward pass (device-only)

Usage:
  python test/dfa_pcc.py
  python test/dfa_pcc.py --ckpt ckpt/latest.pth
"""

import argparse
import os
import sys

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.expanduser("~/project/tt-metal")
)
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/project/Sparse4D"))

import torch
import torch.nn.functional as F
import ttnn

from model.deformable_feature_aggregation import (
    DeformableFeatureAggregation,
    preprocess_dfa_parameters,
)


# ============================================================
# Comparison Utilities
# ============================================================

def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson Correlation Coefficient."""
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == 0 and b.norm() == 0 else 0.0
    return (torch.dot(a, b) / denom).item()


def compute_relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean relative error: mean(|a-b| / max(|a|, |b|, eps))."""
    a = a.float().flatten()
    b = b.float().flatten()
    denom = torch.max(a.abs(), b.abs()).clamp(min=eps)
    return (((a - b).abs() / denom).mean()).item()


def compare_tensors(
    pt: torch.Tensor,
    tt: torch.Tensor,
    name: str,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    num_samples: int = 10,
):
    """Print comprehensive comparison between PyTorch and TT-NN tensors."""
    pt_f = pt.float()
    tt_f = tt.float()
    diff = (pt_f - tt_f).abs()

    pcc = compute_pcc(pt, tt)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_err = compute_relative_error(pt, tt)
    allclose = torch.allclose(pt_f, tt_f, atol=atol, rtol=rtol)

    print(f"\n  --- {name} ---")
    print(f"  Shape:     {list(pt.shape)}")
    print(f"  PCC:       {pcc:.6f}")
    print(f"  MaxDiff:   {max_diff:.6f}")
    print(f"  MeanDiff:  {mean_diff:.6f}")
    print(f"  RelErr:    {rel_err:.6f}")
    print(f"  Allclose:  {allclose}  (atol={atol}, rtol={rtol})")

    # Value range
    print(f"  PT range:  [{pt_f.min().item():.4f}, {pt_f.max().item():.4f}]  mean={pt_f.mean().item():.4f}")
    print(f"  TT range:  [{tt_f.min().item():.4f}, {tt_f.max().item():.4f}]  mean={tt_f.mean().item():.4f}")

    # Sample values
    pt_flat = pt_f.flatten()
    tt_flat = tt_f.flatten()
    n = min(num_samples, pt_flat.numel())
    print(f"\n  Sample values (first {n}):")
    print(f"  {'Index':<8} {'PyTorch':>12} {'TT-NN':>12} {'Diff':>12}")
    print(f"  {'-'*7:<8} {'-'*11:>12} {'-'*11:>12} {'-'*11:>12}")
    for i in range(n):
        d = (pt_flat[i] - tt_flat[i]).abs().item()
        print(f"  {i:<8} {pt_flat[i].item():>12.6f} {tt_flat[i].item():>12.6f} {d:>12.6f}")

    # Worst diff locations
    diff_flat = diff.flatten()
    _, worst_indices = diff_flat.topk(min(5, diff_flat.numel()))
    print(f"\n  Top-5 worst diffs:")
    print(f"  {'Index':<8} {'PyTorch':>12} {'TT-NN':>12} {'Diff':>12}")
    print(f"  {'-'*7:<8} {'-'*11:>12} {'-'*11:>12} {'-'*11:>12}")
    for idx in worst_indices:
        i = idx.item()
        d = diff_flat[i].item()
        print(f"  {i:<8} {pt_flat[i].item():>12.6f} {tt_flat[i].item():>12.6f} {d:>12.6f}")

    return {
        "pcc": pcc,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_err": rel_err,
        "allclose": allclose,
    }


# ============================================================
# Model loading & input creation
# ============================================================

def load_sparse4d_model(ckpt_path: str):
    """Load Sparse4D model and return the DeformableFeatureAggregation module."""
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg_path = os.path.join(
        os.path.dirname(ckpt_path),
        "sparse4dv3_temporal_r50_1x8_bs6_256x704.py",
    )
    if not os.path.exists(cfg_path):
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

    model.eval()
    return model


# Anchor indices
X, Y, Z = 0, 1, 2
W, L, H = 3, 4, 5
SIN_YAW, COS_YAW = 6, 7
VX, VY, VZ = 8, 9, 10


def create_dummy_inputs(
    bs: int = 1,
    num_anchor: int = 900,
    num_cams: int = 6,
    embed_dims: int = 256,
    spatial_shapes: list = None,
):
    """Create dummy inputs for DFA testing."""
    if spatial_shapes is None:
        spatial_shapes = [(64, 176), (32, 88), (16, 44), (8, 22)]

    torch.manual_seed(42)

    instance_feature = torch.randn(bs, num_anchor, embed_dims)
    anchor = torch.randn(bs, num_anchor, 11)
    anchor[..., W:H + 1] = torch.randn(bs, num_anchor, 3) * 0.5
    yaw = torch.randn(bs, num_anchor, 1)
    anchor[..., SIN_YAW] = torch.sin(yaw).squeeze(-1)
    anchor[..., COS_YAW] = torch.cos(yaw).squeeze(-1)

    anchor_embed = torch.randn(bs, num_anchor, embed_dims)
    projection_mat = torch.randn(bs, num_cams, 4, 4)
    image_wh = torch.tensor([[704.0, 256.0]]).unsqueeze(0).expand(bs, num_cams, 2).contiguous()

    feature_maps_pt = []
    for h, w in spatial_shapes:
        fm = torch.randn(bs, num_cams, embed_dims, h, w)
        feature_maps_pt.append(fm)

    metas = {
        "projection_mat": projection_mat,
        "image_wh": image_wh,
    }

    return instance_feature, anchor, anchor_embed, feature_maps_pt, metas


# ============================================================
# Standalone PyTorch reference DFA (no mmcv/mmdet dependency)
# ============================================================

class _KPSGenerator(torch.nn.Module):
    """Pure PyTorch SparseBox3DKeyPointsGenerator."""
    def __init__(self, embed_dims=256, num_learnable_pts=6, fix_scale=None):
        super().__init__()
        if fix_scale is None:
            fix_scale = [[0, 0, 0]]
        self.fix_scale = torch.nn.Parameter(
            torch.tensor(fix_scale, dtype=torch.float32), requires_grad=False
        )
        self.num_pts = len(fix_scale) + num_learnable_pts
        self.num_learnable_pts = num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = torch.nn.Linear(embed_dims, num_learnable_pts * 3)

    def forward(self, anchor, instance_feature=None):
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        key_points = self.fix_scale * size
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid() - 0.5
            )
            key_points = torch.cat([key_points, learnable_scale * size], dim=-2)

        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        key_points = key_points + anchor[..., None, [X, Y, Z]]
        return key_points


class _PTDeformableFeatureAggregation(torch.nn.Module):
    """Pure PyTorch DeformableFeatureAggregation (fallback path only)."""
    def __init__(
        self, embed_dims=256, num_groups=8, num_levels=4, num_cams=6,
        num_learnable_pts=6, use_camera_embed=True, residual_mode="cat",
        fix_scale=None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = embed_dims // num_groups
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.residual_mode = residual_mode
        self.use_deformable_func = False

        self.kps_generator = _KPSGenerator(embed_dims, num_learnable_pts, fix_scale)
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = torch.nn.Linear(embed_dims, embed_dims)
        self.proj_drop = torch.nn.Dropout(0.0)

        if use_camera_embed:
            self.camera_encoder = torch.nn.Sequential(
                torch.nn.Linear(12, embed_dims),
                torch.nn.ReLU(inplace=True),
                torch.nn.LayerNorm(embed_dims),
                torch.nn.Linear(embed_dims, embed_dims),
                torch.nn.ReLU(inplace=True),
                torch.nn.LayerNorm(embed_dims),
            )
            self.weights_fc = torch.nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = torch.nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def forward(self, instance_feature, anchor, anchor_embed, feature_maps, metas, **kwargs):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        weights = self._get_weights(instance_feature, anchor_embed, metas)
        features = self.feature_sampling(
            feature_maps, key_points,
            metas["projection_mat"], metas.get("image_wh"),
        )
        features = self.multi_view_level_fusion(features, weights)
        features = features.sum(dim=2)
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1)
            )
            feature = feature[:, :, None] + camera_embed[:, None]
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(bs, num_anchor, self.num_cams, self.num_levels, self.num_pts, self.num_groups)
        )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    def feature_sampling(self, feature_maps, key_points, projection_mat, image_wh=None):
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]
        points_2d = self.project_points(key_points, projection_mat, image_wh)
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)
        features = []
        for fm in feature_maps:
            features.append(F.grid_sample(fm.flatten(end_dim=1), points_2d))
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(0, 4, 1, 2, 5, 3)
        return features

    def multi_view_level_fusion(self, features, weights):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
        return features


def build_pt_dfa():
    """Build standalone PyTorch DFA (no mmcv dependency)."""
    return _PTDeformableFeatureAggregation(
        embed_dims=256, num_groups=8, num_levels=4, num_cams=6,
        num_learnable_pts=6, use_camera_embed=True, residual_mode="cat",
        fix_scale=[
            [0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
            [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45],
            [0, 0, -0.45],
        ],
    )


def preprocess_dfa_parameters_from_pt(pt_model):
    """Extract parameters from standalone PT DFA (same interface as preprocess_dfa_parameters)."""
    params = {}
    params["kps_fix_scale"] = pt_model.kps_generator.fix_scale.data.clone()
    params["kps_learnable_fc_weight"] = pt_model.kps_generator.learnable_fc.weight.data.clone()
    params["kps_learnable_fc_bias"] = pt_model.kps_generator.learnable_fc.bias.data.clone()
    if pt_model.camera_encoder is not None:
        enc = pt_model.camera_encoder
        params["cam_linear1_weight"] = enc[0].weight.data.clone()
        params["cam_linear1_bias"] = enc[0].bias.data.clone()
        params["cam_ln1_weight"] = enc[2].weight.data.clone()
        params["cam_ln1_bias"] = enc[2].bias.data.clone()
        params["cam_linear2_weight"] = enc[3].weight.data.clone()
        params["cam_linear2_bias"] = enc[3].bias.data.clone()
        params["cam_ln2_weight"] = enc[5].weight.data.clone()
        params["cam_ln2_bias"] = enc[5].bias.data.clone()
    params["weights_fc_weight"] = pt_model.weights_fc.weight.data.clone()
    params["weights_fc_bias"] = pt_model.weights_fc.bias.data.clone()
    params["output_proj_weight"] = pt_model.output_proj.weight.data.clone()
    params["output_proj_bias"] = pt_model.output_proj.bias.data.clone()
    return params


def pytorch_dfa_forward(pt_model, instance_feature, anchor, anchor_embed, feature_maps, metas):
    """Run PyTorch DeformableFeatureAggregation forward (fallback path)."""
    pt_model.eval()
    with torch.no_grad():
        if hasattr(pt_model, 'use_deformable_func'):
            orig_flag = pt_model.use_deformable_func
            pt_model.use_deformable_func = False
        output = pt_model(
            instance_feature=instance_feature,
            anchor=anchor,
            anchor_embed=anchor_embed,
            feature_maps=feature_maps,
            metas=metas,
        )
        if hasattr(pt_model, 'use_deformable_func'):
            pt_model.use_deformable_func = orig_flag
    return output


# ============================================================
# Test 1: grid_sample per FPN level
# ============================================================

def test_grid_sample_pcc(device):
    """Test ttnn.grid_sample vs torch.grid_sample per FPN level."""
    print("\n" + "=" * 65)
    print("  [Test 1] grid_sample: ttnn vs PyTorch (per FPN level)")
    print("=" * 65)

    spatial_shapes = [(64, 176), (32, 88), (16, 44), (8, 22)]
    num_cams = 6
    embed_dims = 256
    num_anchor = 900
    num_pts = 13

    torch.manual_seed(42)

    grid = torch.randn(num_cams, num_anchor, num_pts, 2).float()
    grid = grid.clamp(-1.0, 1.0)

    all_results = []
    for level_idx, (h, w) in enumerate(spatial_shapes):
        fm_nchw = torch.randn(num_cams, embed_dims, h, w).float()
        pt_out = F.grid_sample(
            fm_nchw, grid, mode="bilinear", align_corners=False, padding_mode="zeros"
        )

        fm_nhwc = fm_nchw.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)
        fm_tt = ttnn.from_torch(
            fm_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        grid_tt = ttnn.from_torch(
            grid.float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        tt_out = ttnn.grid_sample(
            fm_tt, grid_tt, mode="bilinear", align_corners=False, padding_mode="zeros"
        )
        tt_out_torch = ttnn.to_torch(tt_out)

        tt_out_nchw = tt_out_torch.permute(0, 3, 1, 2).contiguous()

        result = compare_tensors(
            pt_out, tt_out_nchw,
            name=f"Level {level_idx} [{num_cams},{embed_dims},{h},{w}]",
        )
        all_results.append(result)

        ttnn.deallocate(fm_tt)
        ttnn.deallocate(grid_tt)
        ttnn.deallocate(tt_out)

    print(f"\n  {'Level':<8} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>10} {'RelErr':>10} {'Allclose':>10}")
    print(f"  {'-'*7:<8} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10}")
    for i, r in enumerate(all_results):
        print(
            f"  {i:<8} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} "
            f"{r['mean_diff']:>10.6f} {r['rel_err']:>10.6f} {str(r['allclose']):>10}"
        )

    avg_pcc = sum(r["pcc"] for r in all_results) / len(all_results)
    print(f"\n  Average PCC: {avg_pcc:.6f}")
    return avg_pcc


# ============================================================
# Test 2: End-to-end DFA (device-only)
# ============================================================

def test_e2e_pcc(device, ckpt_path: str = None, bs: int = 1, num_anchor: int = 900):
    """Test end-to-end DFA: PyTorch vs TT-NN (device-only)."""
    print("\n" + "=" * 65)
    print(f"  [Test 2] End-to-end DFA: PyTorch vs TT-NN (device-only, bs={bs})")
    print("=" * 65)

    spatial_shapes = [(64, 176), (32, 88), (16, 44), (8, 22)]
    num_cams = 6
    embed_dims = 256

    instance_feature, anchor, anchor_embed, feature_maps_pt, metas = (
        create_dummy_inputs(bs, num_anchor, num_cams, embed_dims, spatial_shapes)
    )

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"  Using checkpoint: {ckpt_path}")
        full_model = load_sparse4d_model(ckpt_path)
        pt_dfa = None
        for op in full_model.head.op_config_map.values():
            if hasattr(op, 'module') and hasattr(op.module, 'kps_generator'):
                pt_dfa = op.module
                break
        if pt_dfa is None:
            for attr_name in dir(full_model.head):
                attr = getattr(full_model.head, attr_name)
                if hasattr(attr, 'kps_generator'):
                    pt_dfa = attr
                    break
        if pt_dfa is None:
            print("  ERROR: Could not find DeformableFeatureAggregation in model")
            return 0.0
    else:
        print("  Using random weights (no checkpoint)")
        pt_dfa = build_pt_dfa()
        pt_dfa.eval()

    if ckpt_path and os.path.exists(ckpt_path):
        params = preprocess_dfa_parameters(pt_dfa)
    else:
        params = preprocess_dfa_parameters_from_pt(pt_dfa)

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    }

    tt_dfa = DeformableFeatureAggregation(
        device=device,
        parameters=params,
        model_config=model_config,
        embed_dims=256,
        num_groups=8,
        num_levels=4,
        num_cams=6,
        num_pts=13,
        num_learnable_pts=6,
        use_camera_embed=True,
        residual_mode="cat",
    )

    # PyTorch forward
    with torch.no_grad():
        pt_output = pytorch_dfa_forward(
            pt_dfa, instance_feature, anchor, anchor_embed,
            feature_maps_pt, metas,
        )

    # Prepare inputs as ttnn tensors on device
    instance_feature_tt = ttnn.from_torch(
        instance_feature.float(), layout=ttnn.TILE_LAYOUT, device=device
    )
    anchor_tt = ttnn.from_torch(
        anchor.float(), layout=ttnn.TILE_LAYOUT, device=device
    )
    anchor_embed_tt = ttnn.from_torch(
        anchor_embed.float(), layout=ttnn.TILE_LAYOUT, device=device
    )
    projection_mat_tt = ttnn.from_torch(
        metas["projection_mat"].float(), layout=ttnn.TILE_LAYOUT, device=device
    )
    image_wh_tt = ttnn.from_torch(
        metas["image_wh"].float(), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Prepare feature maps for TT
    feature_maps_tt = []
    for level_idx, (h, w) in enumerate(spatial_shapes):
        fm_pt = feature_maps_pt[level_idx]
        fm_flat = fm_pt.reshape(bs * num_cams, embed_dims, h, w)
        fm_nhwc = fm_flat.permute(0, 2, 3, 1).contiguous()
        fm_ttnn_format = fm_nhwc.reshape(1, 1, bs * num_cams * h * w, embed_dims)
        fm_tt = ttnn.from_torch(
            fm_ttnn_format.float(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        feature_maps_tt.append(fm_tt)

    # TT forward (device-only)
    with torch.no_grad():
        tt_output = tt_dfa.run(
            instance_feature_tt,
            anchor_tt,
            anchor_embed_tt,
            feature_maps_tt,
            projection_mat_tt,
            image_wh_tt,
            spatial_shapes,
            bs=bs,
            num_anchor=num_anchor,
        )

    # Convert output to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)
    # Trim padding if needed (TILE may pad)
    if tt_output_torch.shape != pt_output.shape:
        tt_output_torch = tt_output_torch[:bs, :num_anchor, :pt_output.shape[-1]]

    # Full comparison
    result = compare_tensors(
        pt_output, tt_output_torch, name=f"DFA E2E Output (device-only, bs={bs})",
    )

    for fm_tt in feature_maps_tt:
        ttnn.deallocate(fm_tt)

    return result["pcc"]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeformableFeatureAggregation PCC Test (device-only)"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to Sparse4D checkpoint (default: random weights)",
    )
    parser.add_argument(
        "--skip-grid-sample", action="store_true",
        help="Skip grid_sample-only test",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("DeformableFeatureAggregation PCC Test: PyTorch vs TT-NN (device-only)")
    print("=" * 65)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        compute_grid = device.compute_with_storage_grid_size()
        print(
            f"  Device: {device.arch()}, "
            f"grid: {compute_grid.x}x{compute_grid.y}"
        )

        gs_pcc = None
        if not args.skip_grid_sample:
            gs_pcc = test_grid_sample_pcc(device)

        # bs=1: standard inference batch
        e2e_pcc_bs1 = test_e2e_pcc(device, args.ckpt, bs=1, num_anchor=900)

        # bs=2: multi-batch to verify slice+concat rearrange logic
        e2e_pcc_bs2 = test_e2e_pcc(device, args.ckpt, bs=2, num_anchor=900)

        print(f"\n{'=' * 65}")
        print("Final Summary:")
        if gs_pcc is not None:
            print(f"  grid_sample avg PCC:     {gs_pcc:.6f}")
        print(f"  End-to-end DFA PCC bs=1: {e2e_pcc_bs1:.6f}")
        print(f"  End-to-end DFA PCC bs=2: {e2e_pcc_bs2:.6f}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {str(e)[:100]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
