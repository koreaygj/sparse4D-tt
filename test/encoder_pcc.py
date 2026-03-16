"""
SparseBox3DEncoder PCC Test: PyTorch vs TT-NN

Compares the TT-NN SparseBox3DEncoder against PyTorch for numerical accuracy.

Tests:
  1. Cat mode (decouple_attn=True): embed_dims=[128,32,32,64], output=256
  2. Add mode: embed_dims=256, output=256

Usage:
  python test/encoder_pcc.py
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

from model.sparse_box3d_encoder import (
    SparseBox3DEncoder,
    preprocess_encoder_parameters,
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

def _linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    """Replicate mmcv linear_relu_ln."""
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


class _PTSparseBox3DEncoder(nn.Module):
    """Standalone PyTorch SparseBox3DEncoder."""

    X, Y, Z = 0, 1, 2
    W, L, H = 3, 4, 5
    SIN_YAW, COS_YAW = 6, 7
    VX, VY, VZ = 8, 9, 10

    def __init__(
        self,
        embed_dims,
        vel_dims=3,
        mode="cat",
        output_fc=False,
        in_loops=1,
        out_loops=4,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims
        self.mode = mode

        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *_linear_relu_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(2, embed_dims[2])
        if vel_dims > 0:
            self.vel_fc = embedding_layer(vel_dims, embed_dims[3])
        self.output_fc = None
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])

    def forward(self, box_3d):
        pos_feat = self.pos_fc(box_3d[..., [self.X, self.Y, self.Z]])
        size_feat = self.size_fc(box_3d[..., [self.W, self.L, self.H]])
        yaw_feat = self.yaw_fc(box_3d[..., [self.SIN_YAW, self.COS_YAW]])
        if self.mode == "add":
            output = pos_feat + size_feat + yaw_feat
        elif self.mode == "cat":
            output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., self.VX: self.VX + self.vel_dims])
            if self.mode == "add":
                output = output + vel_feat
            elif self.mode == "cat":
                output = torch.cat([output, vel_feat], dim=-1)
        if self.output_fc is not None:
            output = self.output_fc(output)
        return output


def preprocess_encoder_parameters_from_pt(pt_encoder):
    """Extract params from standalone PT encoder."""
    params = {}
    params["pos_fc"] = _extract_linear_relu_ln_params(list(pt_encoder.pos_fc.children()))
    params["size_fc"] = _extract_linear_relu_ln_params(list(pt_encoder.size_fc.children()))
    params["yaw_fc"] = _extract_linear_relu_ln_params(list(pt_encoder.yaw_fc.children()))
    if hasattr(pt_encoder, 'vel_fc'):
        params["vel_fc"] = _extract_linear_relu_ln_params(list(pt_encoder.vel_fc.children()))
    if pt_encoder.output_fc is not None:
        params["output_fc"] = _extract_linear_relu_ln_params(
            list(pt_encoder.output_fc.children())
        )
    return params


# ============================================================
# Test 1: Cat mode (Sparse4D default with decouple_attn)
# ============================================================

def test_cat_mode(device, bs=1, num_anchor=900):
    embed_dims = [128, 32, 32, 64]
    print("\n" + "=" * 65)
    print(f"  [Test 1] Cat mode, dims={embed_dims}, bs={bs}, anchors={num_anchor}")
    print("=" * 65)

    torch.manual_seed(42)

    pt_encoder = _PTSparseBox3DEncoder(
        embed_dims=embed_dims, vel_dims=3, mode="cat",
        output_fc=False, in_loops=1, out_loops=4,
    )
    pt_encoder.eval()

    box_3d = torch.randn(bs, num_anchor, 11)

    with torch.no_grad():
        pt_output = pt_encoder(box_3d)

    params = preprocess_encoder_parameters_from_pt(pt_encoder)
    tt_encoder = SparseBox3DEncoder(
        device, params,
        embed_dims=embed_dims, vel_dims=3, mode="cat",
        has_output_fc=False, in_loops=1, out_loops=4,
    )

    box_3d_tt = ttnn.from_torch(box_3d.float(), layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = tt_encoder.run(box_3d_tt, bs=bs, num_anchor=num_anchor)

    tt_output_torch = ttnn.to_torch(tt_output)
    if tt_output_torch.shape != pt_output.shape:
        tt_output_torch = tt_output_torch[:bs, :num_anchor, :pt_output.shape[-1]]

    result = compare_tensors(pt_output, tt_output_torch, f"Encoder cat mode (dims={embed_dims})")
    return result["pcc"]


# ============================================================
# Test 2: Add mode
# ============================================================

def test_add_mode(device, bs=1, num_anchor=900):
    embed_dims = 256
    print("\n" + "=" * 65)
    print(f"  [Test 2] Add mode, dims={embed_dims}, bs={bs}, anchors={num_anchor}")
    print("=" * 65)

    torch.manual_seed(123)

    pt_encoder = _PTSparseBox3DEncoder(
        embed_dims=embed_dims, vel_dims=3, mode="add",
        output_fc=True, in_loops=1, out_loops=2,
    )
    pt_encoder.eval()

    box_3d = torch.randn(bs, num_anchor, 11)

    with torch.no_grad():
        pt_output = pt_encoder(box_3d)

    params = preprocess_encoder_parameters_from_pt(pt_encoder)
    tt_encoder = SparseBox3DEncoder(
        device, params,
        embed_dims=embed_dims, vel_dims=3, mode="add",
        has_output_fc=True, in_loops=1, out_loops=2,
    )

    box_3d_tt = ttnn.from_torch(box_3d.float(), layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = tt_encoder.run(box_3d_tt, bs=bs, num_anchor=num_anchor)

    tt_output_torch = ttnn.to_torch(tt_output)
    if tt_output_torch.shape != pt_output.shape:
        tt_output_torch = tt_output_torch[:bs, :num_anchor, :pt_output.shape[-1]]

    result = compare_tensors(pt_output, tt_output_torch, f"Encoder add mode (dims={embed_dims})")
    return result["pcc"]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("SparseBox3DEncoder PCC Test: PyTorch vs TT-NN")
    print("=" * 65)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        compute_grid = device.compute_with_storage_grid_size()
        print(f"  Device: {device.arch()}, grid: {compute_grid.x}x{compute_grid.y}")

        pcc_cat = test_cat_mode(device, bs=1, num_anchor=900)
        pcc_add = test_add_mode(device, bs=1, num_anchor=900)

        print(f"\n{'=' * 65}")
        print("Final Summary:")
        print(f"  Cat mode PCC:  {pcc_cat:.6f}")
        print(f"  Add mode PCC:  {pcc_add:.6f}")
        print(f"{'=' * 65}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {str(e)[:200]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
