"""
Sparse4DHead PCC Test with Real Checkpoint Weights

Loads actual trained weights from checkpoint into both PyTorch and TT-NN
heads, runs with synthetic feature maps (same values for both), and compares
step-by-step PCC.

This tests whether PCC degradation is due to random weights (badly conditioned)
or a fundamental TT hardware precision issue.

Usage:
  python test/sparse4d_head_realweight_pcc.py
  python test/sparse4d_head_realweight_pcc.py --ckpt ckpt/latest.pth
"""

import os
import sys
import time
import argparse

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.expanduser("~/project/tt-metal"))
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ttnn

from model.sparse4d_head import Sparse4DHead

# Import the standalone PyTorch modules from existing PCC test
from test.sparse4d_head_pcc import (
    _PTSparse4DHead, _linear_relu_ln,
    compute_pcc, quick_compare, trim,
)

# Same operation order as the real model
OPERATION_ORDER = [
    "deformable", "ffn", "norm", "refine",
    "temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine",
    "temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine",
    "temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine",
    "temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine",
    "temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine",
]

# head.layers.{i} prefix mapping from state_dict
# The PyTorch original uses mmcv modules; we load into our standalone modules.


def _load_mha_from_sd(sd, prefix, pt_mha):
    """Load MHA weights from state_dict into _PTMultiheadAttention."""
    attn = pt_mha.attn
    attn.in_proj_weight.data.copy_(sd[f"{prefix}.attn.in_proj_weight"])
    attn.in_proj_bias.data.copy_(sd[f"{prefix}.attn.in_proj_bias"])
    attn.out_proj.weight.data.copy_(sd[f"{prefix}.attn.out_proj.weight"])
    attn.out_proj.bias.data.copy_(sd[f"{prefix}.attn.out_proj.bias"])


def _load_norm_from_sd(sd, prefix, pt_norm):
    """Load LayerNorm weights from state_dict."""
    pt_norm.weight.data.copy_(sd[f"{prefix}.weight"])
    pt_norm.bias.data.copy_(sd[f"{prefix}.bias"])


def _load_sequential_from_sd(sd, prefix, pt_seq):
    """Load nn.Sequential(Linear, ReLU, LN, ..., Linear) from state_dict."""
    # Map state_dict keys to sequential indices
    # The state_dict uses {prefix}.{idx}.weight/bias
    for name, param in pt_seq.named_parameters():
        key = f"{prefix}.{name}"
        if key in sd:
            param.data.copy_(sd[key])
        else:
            # Try without the module prefix
            pass


def _load_ffn_from_sd(sd, prefix, pt_ffn):
    """Load AsymmetricFFN weights from state_dict."""
    pt_ffn.pre_norm.weight.data.copy_(sd[f"{prefix}.pre_norm.weight"])
    pt_ffn.pre_norm.bias.data.copy_(sd[f"{prefix}.pre_norm.bias"])

    # layers[0][0] = Linear(512, 1024)
    pt_ffn.layers[0][0].weight.data.copy_(sd[f"{prefix}.layers.0.0.weight"])
    pt_ffn.layers[0][0].bias.data.copy_(sd[f"{prefix}.layers.0.0.bias"])
    # layers[1] = Linear(1024, 256)
    pt_ffn.layers[1].weight.data.copy_(sd[f"{prefix}.layers.1.weight"])
    pt_ffn.layers[1].bias.data.copy_(sd[f"{prefix}.layers.1.bias"])

    # identity_fc
    if hasattr(pt_ffn.identity_fc, 'weight'):
        pt_ffn.identity_fc.weight.data.copy_(sd[f"{prefix}.identity_fc.weight"])
        pt_ffn.identity_fc.bias.data.copy_(sd[f"{prefix}.identity_fc.bias"])


def _load_dfa_from_sd(sd, prefix, pt_dfa):
    """Load DeformableFeatureAggregation weights from state_dict."""
    pt_dfa.kps_generator.fix_scale.data.copy_(sd[f"{prefix}.kps_generator.fix_scale"])
    pt_dfa.kps_generator.learnable_fc.weight.data.copy_(sd[f"{prefix}.kps_generator.learnable_fc.weight"])
    pt_dfa.kps_generator.learnable_fc.bias.data.copy_(sd[f"{prefix}.kps_generator.learnable_fc.bias"])

    # camera_encoder: Linear(12,256), ReLU, LN(256), Linear(256,256), ReLU, LN(256)
    enc = pt_dfa.camera_encoder
    enc[0].weight.data.copy_(sd[f"{prefix}.camera_encoder.0.weight"])
    enc[0].bias.data.copy_(sd[f"{prefix}.camera_encoder.0.bias"])
    enc[2].weight.data.copy_(sd[f"{prefix}.camera_encoder.2.weight"])
    enc[2].bias.data.copy_(sd[f"{prefix}.camera_encoder.2.bias"])
    enc[3].weight.data.copy_(sd[f"{prefix}.camera_encoder.3.weight"])
    enc[3].bias.data.copy_(sd[f"{prefix}.camera_encoder.3.bias"])
    enc[5].weight.data.copy_(sd[f"{prefix}.camera_encoder.5.weight"])
    enc[5].bias.data.copy_(sd[f"{prefix}.camera_encoder.5.bias"])

    pt_dfa.weights_fc.weight.data.copy_(sd[f"{prefix}.weights_fc.weight"])
    pt_dfa.weights_fc.bias.data.copy_(sd[f"{prefix}.weights_fc.bias"])
    pt_dfa.output_proj.weight.data.copy_(sd[f"{prefix}.output_proj.weight"])
    pt_dfa.output_proj.bias.data.copy_(sd[f"{prefix}.output_proj.bias"])


def _load_refine_from_sd(sd, prefix, pt_refine):
    """Load SparseBox3DRefinementModule weights from state_dict."""
    # refine layers
    for name, param in pt_refine.layers.named_parameters():
        key = f"{prefix}.layers.{name}"
        if key in sd:
            param.data.copy_(sd[key])

    # cls layers
    for name, param in pt_refine.cls_layers.named_parameters():
        key = f"{prefix}.cls_layers.{name}"
        if key in sd:
            param.data.copy_(sd[key])

    # quality layers
    for name, param in pt_refine.quality_layers.named_parameters():
        key = f"{prefix}.quality_layers.{name}"
        if key in sd:
            param.data.copy_(sd[key])


def _load_encoder_from_sd(sd, prefix, pt_enc):
    """Load SparseBox3DEncoder weights from state_dict."""
    for fc_name in ["pos_fc", "size_fc", "yaw_fc", "vel_fc"]:
        fc = getattr(pt_enc, fc_name, None)
        if fc is None:
            continue
        for name, param in fc.named_parameters():
            key = f"{prefix}.{fc_name}.{name}"
            if key in sd:
                param.data.copy_(sd[key])

    if pt_enc.output_fc is not None:
        for name, param in pt_enc.output_fc.named_parameters():
            key = f"{prefix}.output_fc.{name}"
            if key in sd:
                param.data.copy_(sd[key])


def load_pt_head_from_checkpoint(ckpt_path):
    """Load real checkpoint weights into standalone _PTSparse4DHead."""
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    print(f"  State dict: {len(sd)} parameters")

    # Build PyTorch head with same architecture
    pt_head = _PTSparse4DHead(
        embed_dims=256, num_decoder=6, num_single_frame=1,
        num_anchor=900, num_temp=600, num_cls=10,
        num_groups=8, decouple_attn=True,
    )

    # Load instance bank data
    pt_head.anchor.data.copy_(sd["head.instance_bank.anchor"])
    pt_head.instance_feature.data.copy_(sd["head.instance_bank.instance_feature"])

    # Load fc_before / fc_after
    pt_head.fc_before.weight.data.copy_(sd["head.fc_before.weight"])
    pt_head.fc_after.weight.data.copy_(sd["head.fc_after.weight"])

    # Load each layer
    for i, op in enumerate(OPERATION_ORDER):
        prefix = f"head.layers.{i}"
        layer = pt_head.layers[i]
        if op in ("gnn", "temp_gnn"):
            _load_mha_from_sd(sd, prefix, layer)
        elif op == "norm":
            _load_norm_from_sd(sd, prefix, layer)
        elif op == "ffn":
            _load_ffn_from_sd(sd, prefix, layer)
        elif op == "deformable":
            _load_dfa_from_sd(sd, prefix, layer)
        elif op == "refine":
            _load_refine_from_sd(sd, prefix, layer)

    # Load anchor encoder
    _load_encoder_from_sd(sd, "head.anchor_encoder", pt_head.anchor_encoder)

    pt_head.eval()
    print(f"  Loaded all head weights into PyTorch model")
    return pt_head, sd


def extract_tt_params_from_pt_head(pt_head):
    """Extract TT-NN parameters from the loaded PyTorch head."""
    from test.sparse4d_head_pcc import (
        _extract_mha_params, _extract_norm_params, _extract_ffn_params,
        _extract_dfa_params, _extract_refine_params, _extract_encoder_params,
        extract_all_params,
    )
    return extract_all_params(pt_head)


def test_with_real_weights(device, ckpt_path, bs=1):
    embed_dims = 256
    num_cams = 6
    num_anchor = 900
    spatial_shapes = [(64, 176), (32, 88), (16, 44), (8, 22)]

    print("\n" + "=" * 70)
    print(f"  Sparse4DHead PCC Test with REAL Checkpoint Weights (bs={bs})")
    print("=" * 70)

    # 1. Load real weights into PyTorch head
    pt_head, sd = load_pt_head_from_checkpoint(ckpt_path)

    # 2. Create synthetic but realistic feature maps
    # Use small values typical of FPN output (not random noise)
    torch.manual_seed(42)
    feature_maps_pt = []
    for h, w in spatial_shapes:
        fm = torch.randn(bs, num_cams, embed_dims, h, w) * 0.01
        feature_maps_pt.append(fm)

    metas = {
        "projection_mat": torch.randn(bs, num_cams, 4, 4) * 0.1,
        "image_wh": torch.tensor([[704.0, 256.0]]).unsqueeze(0).expand(bs, num_cams, 2).contiguous(),
        "timestamp": torch.zeros(bs),
    }

    # 3. PyTorch forward
    print("  Running PyTorch forward...", end=" ", flush=True)
    t0 = time.time()
    with torch.no_grad():
        pt_out = pt_head(feature_maps_pt, metas, bs=bs, debug=True)
    print(f"done ({time.time() - t0:.1f}s)")

    # 4. Build TT head and run
    print("  Building TT-NN model...", end=" ", flush=True)
    t0 = time.time()
    params = extract_tt_params_from_pt_head(pt_head)
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

    # Prepare feature maps for TT
    feature_maps_tt = []
    for level_idx, (h, w) in enumerate(spatial_shapes):
        fm_pt = feature_maps_pt[level_idx]
        fm_flat = fm_pt.reshape(bs * num_cams, embed_dims, h, w)
        fm_nhwc = fm_flat.permute(0, 2, 3, 1).contiguous()
        fm_ttnn = fm_nhwc.reshape(1, 1, bs * num_cams * h * w, embed_dims)
        fm_tt = ttnn.from_torch(fm_ttnn.float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32)
        feature_maps_tt.append(fm_tt)

    print("  Running TT-NN forward...", end=" ", flush=True)
    t0 = time.time()
    tt_out = tt_head.forward(feature_maps_tt, metas, bs=bs, debug=True)
    print(f"done ({time.time() - t0:.1f}s)")

    # 5. Compare step by step
    pt_debug = pt_out.get("debug", {})
    tt_debug = tt_out.get("debug", {})

    if pt_debug and tt_debug:
        print(f"\n  === Step-by-step Debug ===")
        print(f"  {'Step':<30} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>10}")
        print(f"  {'-'*29:<30} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10}")

        for key in sorted(pt_debug.keys()):
            if key in tt_debug:
                pt_val = pt_debug[key].float()
                tt_val = tt_debug[key].float()
                if pt_val.shape != tt_val.shape:
                    slices = tuple(slice(0, s) for s in pt_val.shape)
                    tt_val = tt_val[slices]
                r = quick_compare(pt_val, tt_val)
                print(f"  {key:<30} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} {r['mean_diff']:>10.6f}")
        print()

    # Compare outputs
    print(f"  {'Output':<25} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>10}")
    print(f"  {'-'*24:<25} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10}")

    results = {}
    for dec_idx in range(len(pt_out["prediction"])):
        pt_pred = pt_out["prediction"][dec_idx]
        tt_pred = tt_out["prediction"][dec_idx]
        if tt_pred is not None and pt_pred is not None:
            tt_pred_pt = trim(tt_pred, pt_pred.shape)
            r = quick_compare(pt_pred, tt_pred_pt)
            name = f"prediction[{dec_idx}]"
            results[name] = r
            print(f"  {name:<25} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} {r['mean_diff']:>10.6f}")

    for dec_idx in range(len(pt_out["classification"])):
        pt_cls = pt_out["classification"][dec_idx]
        tt_cls = tt_out["classification"][dec_idx]
        if pt_cls is not None and tt_cls is not None:
            tt_cls_pt = trim(tt_cls, pt_cls.shape)
            r = quick_compare(pt_cls, tt_cls_pt)
            name = f"cls[{dec_idx}]"
            results[name] = r
            print(f"  {name:<25} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} {r['mean_diff']:>10.6f}")

    for dec_idx in range(len(pt_out["quality"])):
        pt_qt = pt_out["quality"][dec_idx]
        tt_qt = tt_out["quality"][dec_idx]
        if pt_qt is not None and tt_qt is not None:
            tt_qt_pt = trim(tt_qt, pt_qt.shape)
            r = quick_compare(pt_qt, tt_qt_pt)
            name = f"quality[{dec_idx}]"
            results[name] = r
            print(f"  {name:<25} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} {r['mean_diff']:>10.6f}")

    if results:
        avg_pcc = sum(r["pcc"] for r in results.values()) / len(results)
        all_pass = all(r["pcc"] >= 0.99 for r in results.values())
        print(f"\n  Average PCC: {avg_pcc:.6f}  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    else:
        avg_pcc = 0.0
        print("\n  No results to compare!")

    # Compare with random weight results
    print(f"\n  === Comparison ===")
    print(f"  Random weights avg PCC:  0.959729")
    print(f"  Real weights avg PCC:    {avg_pcc:.6f}")
    if avg_pcc > 0.97:
        print(f"  → Real weights significantly better! Random weights were poorly conditioned.")
    elif avg_pcc > 0.96:
        print(f"  → Similar to random weights. Precision issue is hardware-inherent.")
    else:
        print(f"  → Worse or same. Need further investigation.")

    return avg_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpt/latest.pth", help="Checkpoint path")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"  Checkpoint not found: {args.ckpt}")
        return

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        grid = device.compute_with_storage_grid_size()
        print(f"  Device: {device.arch()}, grid: {grid.x}x{grid.y}")

        pcc = test_with_real_weights(device, args.ckpt, bs=1)

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
