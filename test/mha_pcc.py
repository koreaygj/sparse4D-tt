"""
MultiheadAttention PCC Test: PyTorch vs TT-NN

Compares the TT-NN MultiheadAttention against PyTorch's nn.MultiheadAttention
for numerical accuracy.

Tests:
  1. Self-attention (gnn): query = key (900 anchors)
  2. Cross-attention (temp_gnn): query (900) vs key (600 temporal anchors)
  3. Full graph_model flow with decouple_attn (fc_before + MHA + fc_after)

Usage:
  python test/mha_pcc.py
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

from model.multihead_attention import MultiheadAttention, preprocess_mha_parameters


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

class _PTMultiheadAttention(nn.Module):
    """Mimics mmcv MultiheadAttention behavior for testing."""

    def __init__(self, embed_dims=512, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=False
        )
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, query, key=None, value=None, identity=None, **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query

        # mmcv batch_first: transpose to (seq, batch, dim)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        out = self.attn(query=query, key=key, value=value)[0]

        out = out.transpose(0, 1)
        return identity + self.proj_drop(out)


def preprocess_mha_parameters_from_pt(pt_mha: _PTMultiheadAttention) -> dict:
    """Extract MHA parameters from standalone PyTorch model."""
    params = {}
    attn = pt_mha.attn

    in_proj_weight = attn.in_proj_weight.data.clone()
    w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
    params["w_q"] = w_q.t()
    params["w_k"] = w_k.t()
    params["w_v"] = w_v.t()

    in_proj_bias = attn.in_proj_bias.data.clone()
    b_q, b_k, b_v = in_proj_bias.chunk(3, dim=0)
    params["b_q"] = b_q
    params["b_k"] = b_k
    params["b_v"] = b_v

    params["w_out"] = attn.out_proj.weight.data.clone().t()
    params["b_out"] = attn.out_proj.bias.data.clone()

    return params


# ============================================================
# Test 1: Self-attention (gnn)
# ============================================================

def test_self_attention(device, bs=1, num_anchor=900, embed_dims=512, num_heads=8):
    """Self-attention: query = key = value."""
    print("\n" + "=" * 65)
    print(f"  [Test 1] Self-attention (gnn), bs={bs}, anchors={num_anchor}")
    print("=" * 65)

    torch.manual_seed(42)

    pt_mha = _PTMultiheadAttention(embed_dims, num_heads)
    pt_mha.eval()

    query = torch.randn(bs, num_anchor, embed_dims)

    # PyTorch forward (self-attention: key=None, value=None)
    with torch.no_grad():
        pt_output = pt_mha(query)

    # TT-NN forward
    params = preprocess_mha_parameters_from_pt(pt_mha)
    tt_mha = MultiheadAttention(device, params, embed_dims, num_heads)

    query_tt = ttnn.from_torch(
        query.float(), layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_output = tt_mha.run(
        query=query_tt,
        key=query_tt,
        value=query_tt,
        bs=bs,
        num_queries=num_anchor,
        num_keys=num_anchor,
    )

    tt_output_torch = ttnn.to_torch(tt_output)
    if tt_output_torch.shape != pt_output.shape:
        tt_output_torch = tt_output_torch[:bs, :num_anchor, :embed_dims]

    result = compare_tensors(pt_output, tt_output_torch, "Self-attention output")
    return result["pcc"]


# ============================================================
# Test 2: Cross-attention (temp_gnn)
# ============================================================

def test_cross_attention(
    device, bs=1, num_queries=900, num_keys=600,
    embed_dims=512, num_heads=8,
):
    """Cross-attention: different query and key/value lengths."""
    print("\n" + "=" * 65)
    print(f"  [Test 2] Cross-attention (temp_gnn), bs={bs}, Q={num_queries}, K={num_keys}")
    print("=" * 65)

    torch.manual_seed(123)

    pt_mha = _PTMultiheadAttention(embed_dims, num_heads)
    pt_mha.eval()

    query = torch.randn(bs, num_queries, embed_dims)
    key = torch.randn(bs, num_keys, embed_dims)
    value = torch.randn(bs, num_keys, embed_dims)

    with torch.no_grad():
        pt_output = pt_mha(query, key, value)

    params = preprocess_mha_parameters_from_pt(pt_mha)
    tt_mha = MultiheadAttention(device, params, embed_dims, num_heads)

    query_tt = ttnn.from_torch(query.float(), layout=ttnn.TILE_LAYOUT, device=device)
    key_tt = ttnn.from_torch(key.float(), layout=ttnn.TILE_LAYOUT, device=device)
    value_tt = ttnn.from_torch(value.float(), layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = tt_mha.run(
        query=query_tt,
        key=key_tt,
        value=value_tt,
        bs=bs,
        num_queries=num_queries,
        num_keys=num_keys,
    )

    tt_output_torch = ttnn.to_torch(tt_output)
    if tt_output_torch.shape != pt_output.shape:
        tt_output_torch = tt_output_torch[:bs, :num_queries, :embed_dims]

    result = compare_tensors(pt_output, tt_output_torch, "Cross-attention output")
    return result["pcc"]


# ============================================================
# Test 3: Full graph_model flow (decouple_attn)
# ============================================================

def test_graph_model_flow(device, bs=1, num_anchor=900, embed_dims=256, num_heads=8):
    """Full Sparse4D graph_model flow with decouple_attn=True.

    Flow:
      1. query = cat([instance_feature, anchor_embed], dim=-1)  [bs, N, 512]
      2. value = fc_before(instance_feature)                    [bs, N, 512]
      3. MHA self-attention (key=query)
      4. output = fc_after(mha_output)                          [bs, N, 256]
    """
    print("\n" + "=" * 65)
    print(f"  [Test 3] Full graph_model (decouple_attn), bs={bs}, anchors={num_anchor}")
    print("=" * 65)

    torch.manual_seed(456)
    mha_embed = embed_dims * 2  # 512

    # Build PyTorch modules
    pt_mha = _PTMultiheadAttention(mha_embed, num_heads)
    fc_before = nn.Linear(embed_dims, mha_embed, bias=False)
    fc_after = nn.Linear(mha_embed, embed_dims, bias=False)
    pt_mha.eval()
    fc_before.eval()
    fc_after.eval()

    # Inputs
    instance_feature = torch.randn(bs, num_anchor, embed_dims)
    anchor_embed = torch.randn(bs, num_anchor, embed_dims)

    # PyTorch forward
    with torch.no_grad():
        query = torch.cat([instance_feature, anchor_embed], dim=-1)  # [bs, N, 512]
        value = fc_before(instance_feature)  # [bs, N, 512]
        mha_out = pt_mha(query, key=None, value=value)  # [bs, N, 512]
        pt_output = fc_after(mha_out)  # [bs, N, 256]

    # TT-NN forward
    mha_params = preprocess_mha_parameters_from_pt(pt_mha)
    tt_mha = MultiheadAttention(device, mha_params, mha_embed, num_heads)

    # fc_before/after weights on device
    fc_before_w = ttnn.from_torch(
        fc_before.weight.data.t().float(), layout=ttnn.TILE_LAYOUT, device=device
    )
    fc_after_w = ttnn.from_torch(
        fc_after.weight.data.t().float(), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Inputs on device
    inst_tt = ttnn.from_torch(
        instance_feature.float(), layout=ttnn.TILE_LAYOUT, device=device
    )
    anc_embed_tt = ttnn.from_torch(
        anchor_embed.float(), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Step 1: cat([instance_feature, anchor_embed])
    query_tt = ttnn.concat([inst_tt, anc_embed_tt], dim=-1)  # [bs, N, 512]

    # Step 2: fc_before(instance_feature)
    inst_flat = ttnn.reshape(inst_tt, (1, 1, bs * num_anchor, embed_dims))
    value_tt = ttnn.linear(inst_flat, fc_before_w)  # [1, 1, bs*N, 512]
    value_tt = ttnn.reshape(value_tt, (bs, num_anchor, mha_embed))

    # Step 3: MHA
    mha_out_tt = tt_mha.run(
        query=query_tt,
        key=query_tt,
        value=value_tt,
        bs=bs,
        num_queries=num_anchor,
        num_keys=num_anchor,
    )

    # Step 4: fc_after
    mha_out_flat = ttnn.reshape(mha_out_tt, (1, 1, bs * num_anchor, mha_embed))
    output_tt = ttnn.linear(mha_out_flat, fc_after_w)  # [1, 1, bs*N, 256]
    output_tt = ttnn.reshape(output_tt, (bs, num_anchor, embed_dims))

    tt_output_torch = ttnn.to_torch(output_tt)
    if tt_output_torch.shape != pt_output.shape:
        tt_output_torch = tt_output_torch[:bs, :num_anchor, :embed_dims]

    result = compare_tensors(pt_output, tt_output_torch, "graph_model output (decouple_attn)")
    return result["pcc"]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("MultiheadAttention PCC Test: PyTorch vs TT-NN")
    print("=" * 65)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        compute_grid = device.compute_with_storage_grid_size()
        print(
            f"  Device: {device.arch()}, "
            f"grid: {compute_grid.x}x{compute_grid.y}"
        )

        pcc_self = test_self_attention(device, bs=1, num_anchor=900)
        pcc_cross = test_cross_attention(device, bs=1, num_queries=900, num_keys=600)
        pcc_graph = test_graph_model_flow(device, bs=1, num_anchor=900)

        print(f"\n{'=' * 65}")
        print("Final Summary:")
        print(f"  Self-attention PCC:       {pcc_self:.6f}")
        print(f"  Cross-attention PCC:      {pcc_cross:.6f}")
        print(f"  graph_model flow PCC:     {pcc_graph:.6f}")
        print(f"{'=' * 65}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {str(e)[:200]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
