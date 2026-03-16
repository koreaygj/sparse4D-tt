"""
SparseBox3DRefinementModule PCC Test: PyTorch vs TT-NN

Compares the TT-NN RefinementModule against PyTorch for numerical accuracy.

Tests:
  1. Full refinement with cls + quality (Sparse4D default)
  2. Refinement without quality estimation

Usage:
  python test/refinement_pcc.py
"""

import os
import sys

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.expanduser("~/project/tt-metal")
)
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import ttnn

from model.refinement_module import (
    SparseBox3DRefinementModule,
    preprocess_refinement_parameters,
    _extract_linear_relu_ln_params,
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


def compare_tensors(
    pt: torch.Tensor,
    tt: torch.Tensor,
    name: str,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    num_samples: int = 10,
):
    pt_f = pt.float()
    tt_f = tt.float()
    diff = (pt_f - tt_f).abs()

    pcc = compute_pcc(pt, tt)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    allclose = torch.allclose(pt_f, tt_f, atol=atol, rtol=rtol)

    print(f"\n  --- {name} ---")
    print(f"  Shape:     {list(pt.shape)}")
    print(f"  PCC:       {pcc:.6f}")
    print(f"  MaxDiff:   {max_diff:.6f}")
    print(f"  MeanDiff:  {mean_diff:.6f}")
    print(f"  Allclose:  {allclose}  (atol={atol}, rtol={rtol})")

    pt_flat = pt_f.flatten()
    tt_flat = tt_f.flatten()
    n = min(num_samples, pt_flat.numel())
    print(f"\n  Sample values (first {n}):")
    print(f"  {'Index':<8} {'PyTorch':>12} {'TT-NN':>12} {'Diff':>12}")
    print(f"  {'-'*7:<8} {'-'*11:>12} {'-'*11:>12} {'-'*11:>12}")
    for i in range(n):
        d = (pt_flat[i] - tt_flat[i]).abs().item()
        print(f"  {i:<8} {pt_flat[i].item():>12.6f} {tt_flat[i].item():>12.6f} {d:>12.6f}")

    diff_flat = diff.flatten()
    _, worst_indices = diff_flat.topk(min(5, diff_flat.numel()))
    print(f"\n  Top-5 worst diffs:")
    print(f"  {'Index':<8} {'PyTorch':>12} {'TT-NN':>12} {'Diff':>12}")
    print(f"  {'-'*7:<8} {'-'*11:>12} {'-'*11:>12} {'-'*11:>12}")
    for idx in worst_indices:
        i = idx.item()
        d = diff_flat[i].item()
        print(f"  {i:<8} {pt_flat[i].item():>12.6f} {tt_flat[i].item():>12.6f} {d:>12.6f}")

    return {"pcc": pcc, "max_diff": max_diff, "mean_diff": mean_diff, "allclose": allclose}


# ============================================================
# PyTorch reference (standalone)
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


class _PTRefinementModule(nn.Module):
    """Standalone PyTorch SparseBox3DRefinementModule."""

    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        refine_yaw=True,
        with_quality_estimation=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.refine_yaw = refine_yaw

        self.refine_state = [X, Y, Z, W, L, H]
        if refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.layers = nn.Sequential(
            *_linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(embed_dims, output_dim),
            _Scale([1.0] * output_dim),
        )
        self.cls_layers = nn.Sequential(
            *_linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, num_cls),
        )
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *_linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(embed_dims, 2),
            )

    def forward(self, instance_feature, anchor, anchor_embed, time_interval=1.0, return_cls=True):
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        output[..., self.refine_state] = (
            output[..., self.refine_state] + anchor[..., self.refine_state]
        )
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]

        cls = None
        if return_cls:
            cls = self.cls_layers(instance_feature)

        quality = None
        if return_cls and self.with_quality_estimation:
            quality = self.quality_layers(feature)

        return output, cls, quality


class _Scale(nn.Module):
    """Mimics mmcv Scale."""
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


def preprocess_refinement_parameters_from_pt(pt_module):
    """Extract params from standalone PT refinement module."""
    params = {}

    modules = list(pt_module.layers.children())
    scale_module = modules[-1]
    final_linear = modules[-2]
    chain_modules = modules[:-2]

    params["refine_layers"] = _extract_linear_relu_ln_params(chain_modules)
    params["refine_final_weight"] = final_linear.weight.data.clone().t()
    params["refine_final_bias"] = final_linear.bias.data.clone()
    params["refine_scale"] = scale_module.scale.data.clone()

    cls_modules = list(pt_module.cls_layers.children())
    cls_final = cls_modules[-1]
    cls_chain = cls_modules[:-1]
    params["cls_layers"] = _extract_linear_relu_ln_params(cls_chain)
    params["cls_final_weight"] = cls_final.weight.data.clone().t()
    params["cls_final_bias"] = cls_final.bias.data.clone()

    if hasattr(pt_module, 'quality_layers'):
        qt_modules = list(pt_module.quality_layers.children())
        qt_final = qt_modules[-1]
        qt_chain = qt_modules[:-1]
        params["quality_layers"] = _extract_linear_relu_ln_params(qt_chain)
        params["quality_final_weight"] = qt_final.weight.data.clone().t()
        params["quality_final_bias"] = qt_final.bias.data.clone()

    return params


# ============================================================
# Test 1: Full refinement (cls + quality)
# ============================================================

def test_full_refinement(device, bs=1, num_anchor=900):
    embed_dims = 256
    print("\n" + "=" * 65)
    print(f"  [Test 1] Full refinement (cls+quality), bs={bs}, anchors={num_anchor}")
    print("=" * 65)

    torch.manual_seed(42)

    pt_module = _PTRefinementModule(
        embed_dims=embed_dims, output_dim=11, num_cls=10,
        refine_yaw=True, with_quality_estimation=True,
    )
    pt_module.eval()

    instance_feature = torch.randn(bs, num_anchor, embed_dims)
    anchor = torch.randn(bs, num_anchor, 11)
    anchor_embed = torch.randn(bs, num_anchor, embed_dims)
    time_interval = torch.tensor([0.5] * bs)

    with torch.no_grad():
        pt_output, pt_cls, pt_quality = pt_module(
            instance_feature, anchor, anchor_embed, time_interval
        )

    params = preprocess_refinement_parameters_from_pt(pt_module)
    tt_module = SparseBox3DRefinementModule(
        device, params,
        embed_dims=embed_dims, output_dim=11, num_cls=10,
        refine_yaw=True, with_quality_estimation=True,
    )

    inst_tt = ttnn.from_torch(instance_feature.float(), layout=ttnn.TILE_LAYOUT, device=device)
    anch_tt = ttnn.from_torch(anchor.float(), layout=ttnn.TILE_LAYOUT, device=device)
    embed_tt = ttnn.from_torch(anchor_embed.float(), layout=ttnn.TILE_LAYOUT, device=device)
    ti_tt = ttnn.from_torch(
        time_interval.reshape(1, 1, 1, bs).float(),
        layout=ttnn.TILE_LAYOUT, device=device,
    )
    ti_tt = ttnn.reshape(ti_tt, (bs,))

    tt_output, tt_cls, tt_quality = tt_module.run(
        inst_tt, anch_tt, embed_tt, ti_tt,
        bs=bs, num_anchor=num_anchor, return_cls=True,
    )

    # Compare refined anchor
    tt_out_pt = ttnn.to_torch(tt_output)
    if tt_out_pt.shape != pt_output.shape:
        tt_out_pt = tt_out_pt[:bs, :num_anchor, :11]
    result_anchor = compare_tensors(pt_output, tt_out_pt, "Refined anchor")

    # Compare cls
    tt_cls_pt = ttnn.to_torch(tt_cls)
    if tt_cls_pt.shape != pt_cls.shape:
        tt_cls_pt = tt_cls_pt[:bs, :num_anchor, :10]
    result_cls = compare_tensors(pt_cls, tt_cls_pt, "Classification")

    # Compare quality
    tt_qt_pt = ttnn.to_torch(tt_quality)
    if tt_qt_pt.shape != pt_quality.shape:
        tt_qt_pt = tt_qt_pt[:bs, :num_anchor, :2]
    result_qt = compare_tensors(pt_quality, tt_qt_pt, "Quality")

    return result_anchor["pcc"], result_cls["pcc"], result_qt["pcc"]


# ============================================================
# Test 2: Without quality estimation
# ============================================================

def test_no_quality(device, bs=1, num_anchor=900):
    embed_dims = 256
    print("\n" + "=" * 65)
    print(f"  [Test 2] No quality estimation, bs={bs}, anchors={num_anchor}")
    print("=" * 65)

    torch.manual_seed(123)

    pt_module = _PTRefinementModule(
        embed_dims=embed_dims, output_dim=11, num_cls=10,
        refine_yaw=True, with_quality_estimation=False,
    )
    pt_module.eval()

    instance_feature = torch.randn(bs, num_anchor, embed_dims)
    anchor = torch.randn(bs, num_anchor, 11)
    anchor_embed = torch.randn(bs, num_anchor, embed_dims)
    time_interval = torch.tensor([0.5] * bs)

    with torch.no_grad():
        pt_output, pt_cls, _ = pt_module(
            instance_feature, anchor, anchor_embed, time_interval
        )

    params = preprocess_refinement_parameters_from_pt(pt_module)
    tt_module = SparseBox3DRefinementModule(
        device, params,
        embed_dims=embed_dims, output_dim=11, num_cls=10,
        refine_yaw=True, with_quality_estimation=False,
    )

    inst_tt = ttnn.from_torch(instance_feature.float(), layout=ttnn.TILE_LAYOUT, device=device)
    anch_tt = ttnn.from_torch(anchor.float(), layout=ttnn.TILE_LAYOUT, device=device)
    embed_tt = ttnn.from_torch(anchor_embed.float(), layout=ttnn.TILE_LAYOUT, device=device)
    ti_tt = ttnn.from_torch(
        time_interval.reshape(1, 1, 1, bs).float(),
        layout=ttnn.TILE_LAYOUT, device=device,
    )
    ti_tt = ttnn.reshape(ti_tt, (bs,))

    tt_output, tt_cls, tt_quality = tt_module.run(
        inst_tt, anch_tt, embed_tt, ti_tt,
        bs=bs, num_anchor=num_anchor, return_cls=True,
    )

    tt_out_pt = ttnn.to_torch(tt_output)
    if tt_out_pt.shape != pt_output.shape:
        tt_out_pt = tt_out_pt[:bs, :num_anchor, :11]
    result_anchor = compare_tensors(pt_output, tt_out_pt, "Refined anchor (no quality)")

    tt_cls_pt = ttnn.to_torch(tt_cls)
    if tt_cls_pt.shape != pt_cls.shape:
        tt_cls_pt = tt_cls_pt[:bs, :num_anchor, :10]
    result_cls = compare_tensors(pt_cls, tt_cls_pt, "Classification (no quality)")

    assert tt_quality is None, "Quality should be None"

    return result_anchor["pcc"], result_cls["pcc"]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("SparseBox3DRefinementModule PCC Test: PyTorch vs TT-NN")
    print("=" * 65)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        compute_grid = device.compute_with_storage_grid_size()
        print(f"  Device: {device.arch()}, grid: {compute_grid.x}x{compute_grid.y}")

        pcc_anch, pcc_cls, pcc_qt = test_full_refinement(device, bs=1, num_anchor=900)
        pcc_anch2, pcc_cls2 = test_no_quality(device, bs=1, num_anchor=900)

        print(f"\n{'=' * 65}")
        print("Final Summary:")
        print(f"  [Test 1] Anchor PCC:    {pcc_anch:.6f}")
        print(f"  [Test 1] Cls PCC:       {pcc_cls:.6f}")
        print(f"  [Test 1] Quality PCC:   {pcc_qt:.6f}")
        print(f"  [Test 2] Anchor PCC:    {pcc_anch2:.6f}")
        print(f"  [Test 2] Cls PCC:       {pcc_cls2:.6f}")
        print(f"{'=' * 65}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {str(e)[:200]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
