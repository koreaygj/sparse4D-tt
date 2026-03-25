"""
Sparse4DHead End-to-End PCC Test: PyTorch vs TT-NN

Compares the full Sparse4DHead decoder pipeline (6 layers) between
PyTorch and TT-NN for numerical accuracy.

Since the full model requires mmcv/mmdet3d, this test builds standalone
PyTorch modules that replicate the exact same architecture and weights.

Tests:
  1. Single frame (no temporal): first frame, no cached instances
  2. (Future) Multi-frame with temporal cache

Usage:
  python test/sparse4d_head_pcc.py
"""

import os
import sys
import time

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.expanduser("~/project/tt-metal")
)
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ttnn

from model.sparse4d_head import Sparse4DHead
from model.multihead_attention import preprocess_mha_parameters
from model.asymmetric_ffn import preprocess_ffn_parameters
from model.refinement_module import _extract_linear_relu_ln_params
from model.sparse_box3d_encoder import (
    _extract_linear_relu_ln_params as _extract_enc_params,
)


# ============================================================
# Comparison Utilities
# ============================================================

def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == 0 and b.norm() == 0 else 0.0
    return (torch.dot(a, b) / denom).item()


def quick_compare(pt, tt):
    pt_f, tt_f = pt.float(), tt.float()
    diff = (pt_f - tt_f).abs()
    return {
        "pcc": compute_pcc(pt, tt),
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
    }


def trim(tt_tensor, target_shape):
    t = ttnn.to_torch(tt_tensor)
    slices = tuple(slice(0, s) for s in target_shape)
    return t[slices]


# ============================================================
# Standalone PyTorch modules (no mmcv dependency)
# ============================================================

X, Y, Z = 0, 1, 2
W, L, H = 3, 4, 5
SIN_YAW, COS_YAW = 6, 7
VX, VY, VZ = 8, 9, 10


def _linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class _Scale(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(s, dtype=torch.float32))
    def forward(self, x):
        return x * self.scale


class _PTMultiheadAttention(nn.Module):
    def __init__(self, embed_dims=512, num_heads=8):
        super().__init__()
        self.embed_dims = embed_dims
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=False)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, query, key=None, value=None, identity=None, **kwargs):
        if key is None: key = query
        if value is None: value = key
        if identity is None: identity = query
        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)
        out = self.attn(q, k, v)[0].transpose(0, 1)
        return identity + self.proj_drop(out)


class _PTEncoder(nn.Module):
    def __init__(self, embed_dims, vel_dims=3, mode="cat", output_fc=False, in_loops=1, out_loops=4):
        super().__init__()
        self.mode = mode
        self.vel_dims = vel_dims
        def emb(inp, out):
            return nn.Sequential(*_linear_relu_ln(out, in_loops, out_loops, inp))
        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = emb(3, embed_dims[0])
        self.size_fc = emb(3, embed_dims[1])
        self.yaw_fc = emb(2, embed_dims[2])
        if vel_dims > 0: self.vel_fc = emb(vel_dims, embed_dims[3])
        self.output_fc = emb(embed_dims[-1], embed_dims[-1]) if output_fc else None

    def forward(self, b):
        p = self.pos_fc(b[..., [X, Y, Z]])
        s = self.size_fc(b[..., [W, L, H]])
        y = self.yaw_fc(b[..., [SIN_YAW, COS_YAW]])
        o = torch.cat([p, s, y], dim=-1) if self.mode == "cat" else p + s + y
        if self.vel_dims > 0:
            v = self.vel_fc(b[..., VX:VX + self.vel_dims])
            o = torch.cat([o, v], dim=-1) if self.mode == "cat" else o + v
        if self.output_fc is not None: o = self.output_fc(o)
        return o


class _PTAsymmetricFFN(nn.Module):
    def __init__(self, in_channels=512, embed_dims=256, ff_channels=1024):
        super().__init__()
        self.pre_norm = nn.LayerNorm(in_channels)
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(in_channels, ff_channels), nn.ReLU(inplace=True), nn.Dropout(0.0)),
            nn.Linear(ff_channels, embed_dims), nn.Dropout(0.0),
        )
        self.identity_fc = nn.Linear(in_channels, embed_dims) if in_channels != embed_dims else nn.Identity()

    def forward(self, x, identity=None):
        x = self.pre_norm(x)
        if identity is None: identity = x
        return self.identity_fc(identity) + self.layers(x)


class _PTRefinement(nn.Module):
    def __init__(self, embed_dims=256, output_dim=11, num_cls=10):
        super().__init__()
        self.output_dim = output_dim
        self.refine_state = [X, Y, Z, W, L, H, SIN_YAW, COS_YAW]
        self.layers = nn.Sequential(
            *_linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(embed_dims, output_dim), _Scale([1.0] * output_dim),
        )
        self.cls_layers = nn.Sequential(
            *_linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, num_cls),
        )
        self.quality_layers = nn.Sequential(
            *_linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, 2),
        )

    def forward(self, inst, anch, embed, ti=1.0, return_cls=True):
        feat = inst + embed
        out = self.layers(feat)
        out[..., self.refine_state] += anch[..., self.refine_state]
        if self.output_dim > 8:
            if not isinstance(ti, torch.Tensor): ti = inst.new_tensor(ti)
            tr = torch.transpose(out[..., VX:], 0, -1)
            out[..., VX:] = torch.transpose(tr / ti, 0, -1) + anch[..., VX:]
        cls = self.cls_layers(inst) if return_cls else None
        qt = self.quality_layers(feat) if return_cls else None
        return out, cls, qt


class _PTDeformableFeatureAggregation(nn.Module):
    """Simplified DFA for head-level testing (uses grid_sample)."""
    def __init__(self, embed_dims=256, num_groups=8, num_levels=4, num_cams=6,
                 num_learnable_pts=6, fix_scale=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = embed_dims // num_groups
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.kps_generator = _PTKPSGenerator(embed_dims, num_learnable_pts, fix_scale)
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(0.0)
        self.camera_encoder = nn.Sequential(
            nn.Linear(12, embed_dims), nn.ReLU(inplace=True), nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True), nn.LayerNorm(embed_dims),
        )
        self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts)

    def forward(self, instance_feature, anchor, anchor_embed, feature_maps, metas, **kw):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        weights = self._get_weights(instance_feature, anchor_embed, metas)
        features = self._feature_sampling(feature_maps, key_points,
                                          metas["projection_mat"], metas.get("image_wh"))
        features = self._fusion(features, weights)
        features = features.sum(dim=2)
        output = self.proj_drop(self.output_proj(features))
        return torch.cat([output, instance_feature], dim=-1)  # residual_mode="cat"

    def _get_weights(self, inst, embed, metas):
        bs, na = inst.shape[:2]
        feat = inst + embed
        cam_embed = self.camera_encoder(metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1))
        feat = feat[:, :, None] + cam_embed[:, None]
        w = self.weights_fc(feat).reshape(bs, na, -1, self.num_groups).softmax(dim=-2)
        return w.reshape(bs, na, self.num_cams, self.num_levels, self.num_pts, self.num_groups)

    def _feature_sampling(self, fmaps, kpts, proj, wh=None):
        nl = len(fmaps); nc = fmaps[0].shape[1]
        bs, na, np_ = kpts.shape[:3]
        pts = torch.cat([kpts, torch.ones_like(kpts[..., :1])], dim=-1)
        pts_2d = torch.matmul(proj[:, :, None, None], pts[:, None, ..., None]).squeeze(-1)
        pts_2d = pts_2d[..., :2] / torch.clamp(pts_2d[..., 2:3], min=1e-5)
        if wh is not None: pts_2d = pts_2d / wh[:, :, None, None]
        pts_2d = pts_2d * 2 - 1
        pts_2d = pts_2d.flatten(end_dim=1)
        feats = []
        for fm in fmaps:
            feats.append(F.grid_sample(fm.flatten(end_dim=1), pts_2d))
        feats = torch.stack(feats, dim=1)
        return feats.reshape(bs, nc, nl, -1, na, np_).permute(0, 4, 1, 2, 5, 3)

    def _fusion(self, features, weights):
        bs, na = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims))
        features = features.sum(dim=2).sum(dim=2)
        return features.reshape(bs, na, self.num_pts, self.embed_dims)


class _PTKPSGenerator(nn.Module):
    def __init__(self, embed_dims=256, num_learnable_pts=6, fix_scale=None):
        super().__init__()
        if fix_scale is None:
            fix_scale = [[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                         [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45], [0, 0, -0.45]]
        self.fix_scale = nn.Parameter(torch.tensor(fix_scale, dtype=torch.float32), requires_grad=False)
        self.num_pts = len(fix_scale) + num_learnable_pts
        self.learnable_fc = nn.Linear(embed_dims, num_learnable_pts * 3)
        self.num_learnable_pts = num_learnable_pts

    def forward(self, anchor, instance_feature=None):
        bs, na = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        kp = self.fix_scale * size
        if instance_feature is not None:
            ls = self.learnable_fc(instance_feature).reshape(bs, na, self.num_learnable_pts, 3).sigmoid() - 0.5
            kp = torch.cat([kp, ls * size], dim=-2)
        rot = anchor.new_zeros([bs, na, 3, 3])
        rot[:, :, 0, 0] = anchor[:, :, COS_YAW]; rot[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rot[:, :, 1, 0] = anchor[:, :, SIN_YAW]; rot[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rot[:, :, 2, 2] = 1
        kp = torch.matmul(rot[:, :, None], kp[..., None]).squeeze(-1)
        return kp + anchor[..., None, [X, Y, Z]]


# ============================================================
# PyTorch Sparse4DHead (standalone)
# ============================================================

class _PTSparse4DHead(nn.Module):
    """Standalone PyTorch Sparse4DHead for PCC testing."""

    def __init__(self, embed_dims=256, num_decoder=6, num_single_frame=1,
                 num_anchor=900, num_temp=600, num_cls=10, num_groups=8,
                 decouple_attn=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame
        self.num_anchor = num_anchor
        self.num_temp_instances = num_temp
        self.decouple_attn = decouple_attn
        mha_embed = embed_dims * 2 if decouple_attn else embed_dims

        self.operation_order = (
            ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * num_single_frame
            + ["temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine"]
            * (num_decoder - num_single_frame)
        )[2:]

        layers = []
        for op in self.operation_order:
            if op in ("gnn", "temp_gnn"):
                layers.append(_PTMultiheadAttention(mha_embed, num_groups))
            elif op == "norm":
                layers.append(nn.LayerNorm(embed_dims))
            elif op == "deformable":
                layers.append(_PTDeformableFeatureAggregation(embed_dims, num_groups))
            elif op == "ffn":
                layers.append(_PTAsymmetricFFN(embed_dims * 2, embed_dims, embed_dims * 4))
            elif op == "refine":
                layers.append(_PTRefinement(embed_dims, 11, num_cls))
        self.layers = nn.ModuleList(layers)

        self.anchor_encoder = _PTEncoder([128, 32, 32, 64], mode="cat", output_fc=False, in_loops=1, out_loops=4)

        # Instance bank data
        self.anchor = nn.Parameter(torch.randn(num_anchor, 11) * 0.1)
        self.instance_feature = nn.Parameter(torch.zeros(num_anchor, embed_dims))

        if decouple_attn:
            self.fc_before = nn.Linear(embed_dims, mha_embed, bias=False)
            self.fc_after = nn.Linear(mha_embed, embed_dims, bias=False)

    def graph_model(self, idx, query, key=None, value=None, query_pos=None, key_pos=None):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(self.layers[idx](query, key, value, query_pos=query_pos, key_pos=key_pos))

    def forward(self, feature_maps, metas, bs=1, debug=False):
        num_anchor = self.anchor.shape[0]
        instance_feature = self.instance_feature[None].expand(bs, -1, -1).clone()
        anchor = self.anchor[None].expand(bs, -1, -1).clone()
        time_interval = torch.tensor([0.5] * bs, dtype=torch.float32)

        anchor_embed = self.anchor_encoder(anchor)

        prediction = []
        classification = []
        quality = []
        debug_intermediates = {}

        if debug:
            debug_intermediates["anchor_embed_init"] = anchor_embed.clone()
            debug_intermediates["instance_feature_init"] = instance_feature.clone()

        for i, op in enumerate(self.operation_order):
            if op == "temp_gnn":
                # No temporal data: key=None, value=None
                # PyTorch original passes temp_instance_feature (None)
                # → graph_model skips fc_before, MHA uses query as K and V
                instance_feature = self.graph_model(
                    i, instance_feature, key=None, value=None,
                    query_pos=anchor_embed,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i, instance_feature, value=instance_feature,
                    query_pos=anchor_embed,
                )
            elif op == "norm":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature, anchor, anchor_embed, feature_maps, metas,
                )
            elif op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "refine":
                return_cls = (
                    len(prediction) == self.num_single_frame_decoder - 1
                    or i == len(self.operation_order) - 1
                )
                anchor, cls, qt = self.layers[i](
                    instance_feature, anchor, anchor_embed,
                    time_interval, return_cls,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)

            if debug:
                debug_intermediates[f"step_{i}_{op}"] = instance_feature.clone()
                if op == "refine":
                    debug_intermediates[f"step_{i}_anchor"] = anchor.clone()

        result = {"prediction": prediction, "classification": classification, "quality": quality}
        if debug:
            result["debug"] = debug_intermediates
        return result


# ============================================================
# Parameter extraction: PT → TT-NN
# ============================================================

def _extract_mha_params(pt_mha):
    attn = pt_mha.attn
    iw = attn.in_proj_weight.data.clone()
    wq, wk, wv = iw.chunk(3, dim=0)
    ib = attn.in_proj_bias.data.clone()
    bq, bk, bv = ib.chunk(3, dim=0)
    return {
        "w_q": wq.t(), "w_k": wk.t(), "w_v": wv.t(),
        "b_q": bq, "b_k": bk, "b_v": bv,
        "w_out": attn.out_proj.weight.data.clone().t(),
        "b_out": attn.out_proj.bias.data.clone(),
    }


def _extract_norm_params(pt_norm):
    return {"weight": pt_norm.weight.data.clone(), "bias": pt_norm.bias.data.clone()}


def _extract_ffn_params(pt_ffn):
    return {
        "pre_norm_weight": pt_ffn.pre_norm.weight.data.clone(),
        "pre_norm_bias": pt_ffn.pre_norm.bias.data.clone(),
        "fc1_weight": pt_ffn.layers[0][0].weight.data.clone().t(),
        "fc1_bias": pt_ffn.layers[0][0].bias.data.clone(),
        "fc2_weight": pt_ffn.layers[1].weight.data.clone().t(),
        "fc2_bias": pt_ffn.layers[1].bias.data.clone(),
        "identity_fc_weight": pt_ffn.identity_fc.weight.data.clone().t(),
        "identity_fc_bias": pt_ffn.identity_fc.bias.data.clone(),
    }


def _extract_dfa_params(pt_dfa):
    from model.deformable_feature_aggregation import preprocess_dfa_parameters as _dfa_ext
    params = {}
    params["kps_fix_scale"] = pt_dfa.kps_generator.fix_scale.data.clone()
    params["kps_learnable_fc_weight"] = pt_dfa.kps_generator.learnable_fc.weight.data.clone()
    params["kps_learnable_fc_bias"] = pt_dfa.kps_generator.learnable_fc.bias.data.clone()
    enc = pt_dfa.camera_encoder
    params["cam_linear1_weight"] = enc[0].weight.data.clone()
    params["cam_linear1_bias"] = enc[0].bias.data.clone()
    params["cam_ln1_weight"] = enc[2].weight.data.clone()
    params["cam_ln1_bias"] = enc[2].bias.data.clone()
    params["cam_linear2_weight"] = enc[3].weight.data.clone()
    params["cam_linear2_bias"] = enc[3].bias.data.clone()
    params["cam_ln2_weight"] = enc[5].weight.data.clone()
    params["cam_ln2_bias"] = enc[5].bias.data.clone()
    params["weights_fc_weight"] = pt_dfa.weights_fc.weight.data.clone()
    params["weights_fc_bias"] = pt_dfa.weights_fc.bias.data.clone()
    params["output_proj_weight"] = pt_dfa.output_proj.weight.data.clone()
    params["output_proj_bias"] = pt_dfa.output_proj.bias.data.clone()
    return params


def _extract_refine_params(pt_ref):
    mods = list(pt_ref.layers.children())
    params = {
        "refine_layers": _extract_linear_relu_ln_params(mods[:-2]),
        "refine_final_weight": mods[-2].weight.data.clone().t(),
        "refine_final_bias": mods[-2].bias.data.clone(),
        "refine_scale": mods[-1].scale.data.clone(),
    }
    cls_mods = list(pt_ref.cls_layers.children())
    params["cls_layers"] = _extract_linear_relu_ln_params(cls_mods[:-1])
    params["cls_final_weight"] = cls_mods[-1].weight.data.clone().t()
    params["cls_final_bias"] = cls_mods[-1].bias.data.clone()
    qt_mods = list(pt_ref.quality_layers.children())
    params["quality_layers"] = _extract_linear_relu_ln_params(qt_mods[:-1])
    params["quality_final_weight"] = qt_mods[-1].weight.data.clone().t()
    params["quality_final_bias"] = qt_mods[-1].bias.data.clone()
    return params


def _extract_encoder_params(pt_enc):
    p = {}
    for name in ["pos_fc", "size_fc", "yaw_fc"]:
        p[name] = _extract_enc_params(list(getattr(pt_enc, name).children()))
    if hasattr(pt_enc, 'vel_fc'):
        p["vel_fc"] = _extract_enc_params(list(pt_enc.vel_fc.children()))
    if pt_enc.output_fc is not None:
        p["output_fc"] = _extract_enc_params(list(pt_enc.output_fc.children()))
    return p


def extract_all_params(pt_head):
    """Extract all parameters from _PTSparse4DHead for TT-NN."""
    params = {}

    layer_params = []
    for i, op in enumerate(pt_head.operation_order):
        layer = pt_head.layers[i]
        if op in ("gnn", "temp_gnn"):
            layer_params.append(_extract_mha_params(layer))
        elif op == "norm":
            layer_params.append(_extract_norm_params(layer))
        elif op == "deformable":
            layer_params.append(_extract_dfa_params(layer))
        elif op == "ffn":
            layer_params.append(_extract_ffn_params(layer))
        elif op == "refine":
            layer_params.append(_extract_refine_params(layer))
    params["layers"] = layer_params

    params["anchor_encoder"] = _extract_encoder_params(pt_head.anchor_encoder)
    params["instance_bank"] = {
        "anchor_data": pt_head.anchor.data.clone(),
        "instance_feature_data": pt_head.instance_feature.data.clone(),
    }

    if pt_head.decouple_attn:
        params["fc_before_weight"] = pt_head.fc_before.weight.data.clone().t()
        params["fc_after_weight"] = pt_head.fc_after.weight.data.clone().t()

    return params


# ============================================================
# Test: Single frame (no temporal)
# ============================================================

def test_single_frame(device, bs=1, num_anchor=900):
    embed_dims = 256
    num_cams = 6
    spatial_shapes = [(64, 176), (32, 88), (16, 44), (8, 22)]

    print("\n" + "=" * 70)
    print(f"  Sparse4DHead E2E PCC Test (single frame, bs={bs})")
    print("=" * 70)

    torch.manual_seed(42)

    # Build PyTorch model
    pt_head = _PTSparse4DHead(
        embed_dims=embed_dims, num_decoder=6, num_single_frame=1,
        num_anchor=num_anchor, num_temp=600, num_cls=10,
        num_groups=8, decouple_attn=True,
    )
    pt_head.eval()

    # Create inputs
    feature_maps_pt = []
    for h, w in spatial_shapes:
        fm = torch.randn(bs, num_cams, embed_dims, h, w) * 0.1
        feature_maps_pt.append(fm)

    metas = {
        "projection_mat": torch.randn(bs, num_cams, 4, 4) * 0.1,
        "image_wh": torch.tensor([[704.0, 256.0]]).unsqueeze(0).expand(bs, num_cams, 2).contiguous(),
        "timestamp": torch.zeros(bs),
    }

    # PyTorch forward (with debug)
    print("  Running PyTorch forward...", end=" ", flush=True)
    t0 = time.time()
    with torch.no_grad():
        pt_out = pt_head(feature_maps_pt, metas, bs=bs, debug=True)
    print(f"done ({time.time() - t0:.1f}s)")

    # Extract parameters & build TT-NN model
    print("  Building TT-NN model...", end=" ", flush=True)
    t0 = time.time()
    params = extract_all_params(pt_head)
    tt_head = Sparse4DHead(
        device=device,
        parameters=params,
        operation_order=pt_head.operation_order,
        embed_dims=embed_dims,
        num_decoder=6,
        num_single_frame_decoder=1,
        num_anchor=num_anchor,
        num_temp_instances=600,
        num_classes=10,
        num_groups=8,
        spatial_shapes=spatial_shapes,
    )
    print(f"done ({time.time() - t0:.1f}s)")

    # Prepare feature maps for TT-NN
    feature_maps_tt = []
    for level_idx, (h, w) in enumerate(spatial_shapes):
        fm_pt = feature_maps_pt[level_idx]
        fm_flat = fm_pt.reshape(bs * num_cams, embed_dims, h, w)
        fm_nhwc = fm_flat.permute(0, 2, 3, 1).contiguous()
        fm_ttnn = fm_nhwc.reshape(1, 1, bs * num_cams * h * w, embed_dims)
        fm_tt = ttnn.from_torch(fm_ttnn.float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32)
        feature_maps_tt.append(fm_tt)

    # TT-NN forward (with debug)
    print("  Running TT-NN forward...", end=" ", flush=True)
    t0 = time.time()
    tt_out = tt_head.forward(feature_maps_tt, metas, bs=bs, debug=True)
    print(f"done ({time.time() - t0:.1f}s)")

    # ---- Debug: step-by-step comparison ----
    pt_debug = pt_out.get("debug", {})
    tt_debug = tt_out.get("debug", {})

    if pt_debug and tt_debug:
        print(f"\n  === Step-by-step Debug (1st decoder) ===")
        print(f"  {'Step':<30} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>10}")
        print(f"  {'-'*29:<30} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10}")

        for key in sorted(pt_debug.keys()):
            if key in tt_debug:
                pt_val = pt_debug[key].float()
                tt_val = tt_debug[key].float()
                # Trim if shapes differ (TILE padding)
                if pt_val.shape != tt_val.shape:
                    slices = tuple(slice(0, s) for s in pt_val.shape)
                    tt_val = tt_val[slices]
                r = quick_compare(pt_val, tt_val)
                print(f"  {key:<30} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} {r['mean_diff']:>10.6f}")
        print()

    # Compare results
    print(f"\n  {'Output':<25} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>10}")
    print(f"  {'-'*24:<25} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10}")

    results = {}

    # Compare final prediction (last decoder output)
    for dec_idx in range(len(pt_out["prediction"])):
        pt_pred = pt_out["prediction"][dec_idx]
        tt_pred = tt_out["prediction"][dec_idx]
        if tt_pred is not None and pt_pred is not None:
            tt_pred_pt = trim(tt_pred, pt_pred.shape)
            r = quick_compare(pt_pred, tt_pred_pt)
            name = f"prediction[{dec_idx}]"
            results[name] = r
            print(f"  {name:<25} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} {r['mean_diff']:>10.6f}")

    # Compare classification (only non-None)
    for dec_idx in range(len(pt_out["classification"])):
        pt_cls = pt_out["classification"][dec_idx]
        tt_cls = tt_out["classification"][dec_idx]
        if pt_cls is not None and tt_cls is not None:
            tt_cls_pt = trim(tt_cls, pt_cls.shape)
            r = quick_compare(pt_cls, tt_cls_pt)
            name = f"cls[{dec_idx}]"
            results[name] = r
            print(f"  {name:<25} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} {r['mean_diff']:>10.6f}")

    # Compare quality (only non-None)
    for dec_idx in range(len(pt_out["quality"])):
        pt_qt = pt_out["quality"][dec_idx]
        tt_qt = tt_out["quality"][dec_idx]
        if pt_qt is not None and tt_qt is not None:
            tt_qt_pt = trim(tt_qt, pt_qt.shape)
            r = quick_compare(pt_qt, tt_qt_pt)
            name = f"quality[{dec_idx}]"
            results[name] = r
            print(f"  {name:<25} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} {r['mean_diff']:>10.6f}")

    # Summary
    if results:
        avg_pcc = sum(r["pcc"] for r in results.values()) / len(results)
        all_pass = all(r["pcc"] >= 0.99 for r in results.values())
        print(f"\n  Average PCC: {avg_pcc:.6f}  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    else:
        avg_pcc = 0.0
        print("\n  No results to compare!")

    return avg_pcc


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  Sparse4DHead End-to-End PCC Test: PyTorch vs TT-NN")
    print("=" * 70)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        grid = device.compute_with_storage_grid_size()
        print(f"  Device: {device.arch()}, grid: {grid.x}x{grid.y}")

        pcc = test_single_frame(device, bs=1, num_anchor=900)

        print(f"\n{'=' * 70}")
        print(f"  Final: Average PCC = {pcc:.6f}")
        print(f"{'=' * 70}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  FAILED: {str(e)[:300]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
