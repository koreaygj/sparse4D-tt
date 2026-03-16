"""
AsymmetricFFN PCC Test: PyTorch vs TT-NN

Compares the TT-NN AsymmetricFFN against PyTorch for numerical accuracy.

Tests:
  1. Asymmetric (in=512, out=256): Sparse4D default after DFA cat
  2. Symmetric (in=256, out=256): identity_fc is nn.Identity

Usage:
  python test/ffn_pcc.py
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

from model.asymmetric_ffn import AsymmetricFFN, preprocess_ffn_parameters


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
# PyTorch reference (standalone, no mmcv dependency)
# ============================================================

class _PTAsymmetricFFN(nn.Module):
    """Mimics mmcv AsymmetricFFN for testing."""

    def __init__(
        self,
        in_channels=512,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        pre_norm=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        if pre_norm:
            self.pre_norm = nn.LayerNorm(in_channels)
        else:
            self.pre_norm = None

        # layers: Sequential(Sequential(Linear, ReLU, Dropout), Linear, Dropout)
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(in_channels, feedforward_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.0),
            ),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(0.0),
        )

        if in_channels != embed_dims:
            self.identity_fc = nn.Linear(in_channels, embed_dims)
        else:
            self.identity_fc = nn.Identity()

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x_normed = self.pre_norm(x)
        else:
            x_normed = x
        out = self.layers(x_normed)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + out


def preprocess_ffn_parameters_from_pt(pt_ffn: _PTAsymmetricFFN) -> dict:
    """Extract FFN parameters from standalone PyTorch model."""
    params = {}

    if pt_ffn.pre_norm is not None:
        params["pre_norm_weight"] = pt_ffn.pre_norm.weight.data.clone()
        params["pre_norm_bias"] = pt_ffn.pre_norm.bias.data.clone()

    fc1 = pt_ffn.layers[0][0]
    params["fc1_weight"] = fc1.weight.data.clone().t()
    params["fc1_bias"] = fc1.bias.data.clone()

    fc2 = pt_ffn.layers[1]
    params["fc2_weight"] = fc2.weight.data.clone().t()
    params["fc2_bias"] = fc2.bias.data.clone()

    if not isinstance(pt_ffn.identity_fc, nn.Identity):
        params["identity_fc_weight"] = pt_ffn.identity_fc.weight.data.clone().t()
        params["identity_fc_bias"] = pt_ffn.identity_fc.bias.data.clone()

    return params


# ============================================================
# Test 1: Asymmetric FFN (in=512, out=256)
# ============================================================

def test_asymmetric(device, bs=1, num_tokens=900):
    """Asymmetric: in_channels=512, embed_dims=256 (Sparse4D default)."""
    in_channels = 512
    embed_dims = 256
    feedforward_channels = 1024

    print("\n" + "=" * 65)
    print(f"  [Test 1] Asymmetric FFN ({in_channels}→{embed_dims}), bs={bs}, tokens={num_tokens}")
    print("=" * 65)

    torch.manual_seed(42)

    pt_ffn = _PTAsymmetricFFN(in_channels, embed_dims, feedforward_channels)
    pt_ffn.eval()

    x = torch.randn(bs, num_tokens, in_channels)

    with torch.no_grad():
        pt_output = pt_ffn(x)

    params = preprocess_ffn_parameters_from_pt(pt_ffn)
    tt_ffn = AsymmetricFFN(device, params, in_channels, embed_dims, feedforward_channels)

    x_tt = ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = tt_ffn.run(x=x_tt, bs=bs, num_tokens=num_tokens)

    tt_output_torch = ttnn.to_torch(tt_output)
    if tt_output_torch.shape != pt_output.shape:
        tt_output_torch = tt_output_torch[:bs, :num_tokens, :embed_dims]

    result = compare_tensors(pt_output, tt_output_torch, "Asymmetric FFN (512→256)")
    return result["pcc"]


# ============================================================
# Test 2: Symmetric FFN (in=256, out=256)
# ============================================================

def test_symmetric(device, bs=1, num_tokens=900):
    """Symmetric: in_channels=embed_dims=256 (identity_fc is Identity)."""
    in_channels = 256
    embed_dims = 256
    feedforward_channels = 1024

    print("\n" + "=" * 65)
    print(f"  [Test 2] Symmetric FFN ({in_channels}→{embed_dims}), bs={bs}, tokens={num_tokens}")
    print("=" * 65)

    torch.manual_seed(123)

    pt_ffn = _PTAsymmetricFFN(in_channels, embed_dims, feedforward_channels)
    pt_ffn.eval()

    x = torch.randn(bs, num_tokens, in_channels)

    with torch.no_grad():
        pt_output = pt_ffn(x)

    params = preprocess_ffn_parameters_from_pt(pt_ffn)
    tt_ffn = AsymmetricFFN(device, params, in_channels, embed_dims, feedforward_channels)

    x_tt = ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = tt_ffn.run(x=x_tt, bs=bs, num_tokens=num_tokens)

    tt_output_torch = ttnn.to_torch(tt_output)
    if tt_output_torch.shape != pt_output.shape:
        tt_output_torch = tt_output_torch[:bs, :num_tokens, :embed_dims]

    result = compare_tensors(pt_output, tt_output_torch, "Symmetric FFN (256→256)")
    return result["pcc"]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("AsymmetricFFN PCC Test: PyTorch vs TT-NN")
    print("=" * 65)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        compute_grid = device.compute_with_storage_grid_size()
        print(
            f"  Device: {device.arch()}, "
            f"grid: {compute_grid.x}x{compute_grid.y}"
        )

        pcc_asym = test_asymmetric(device, bs=1, num_tokens=900)
        pcc_sym = test_symmetric(device, bs=1, num_tokens=900)

        print(f"\n{'=' * 65}")
        print("Final Summary:")
        print(f"  Asymmetric FFN (512→256) PCC:  {pcc_asym:.6f}")
        print(f"  Symmetric FFN (256→256) PCC:   {pcc_sym:.6f}")
        print(f"{'=' * 65}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {str(e)[:200]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
