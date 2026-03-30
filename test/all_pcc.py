"""
Sparse4D Module PCC Summary: All modules PyTorch vs TT-NN

Runs all individual module PCC tests and produces a consolidated report.

Modules tested:
  1. MultiheadAttention (self-attn, cross-attn, graph_model flow)
  2. AsymmetricFFN (asymmetric 512→256, symmetric 256→256)
  3. SparseBox3DEncoder (cat mode, add mode)
  4. SparseBox3DRefinementModule (anchor, cls, quality)

Note: ResNet50, FPN, DFA have separate tests (resnet_pcc.py, fpn_pcc.py, dfa_pcc.py)
      and are not included here to keep runtime manageable.

Usage:
  python test/all_pcc.py
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
import ttnn

# ============================================================
# Shared utilities
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


def quick_compare(pt: torch.Tensor, tt: torch.Tensor):
    """Return pcc, max_diff, mean_diff without printing."""
    pt_f, tt_f = pt.float(), tt.float()
    diff = (pt_f - tt_f).abs()
    return {
        "pcc": compute_pcc(pt, tt),
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "allclose": torch.allclose(pt_f, tt_f, atol=1e-2, rtol=1e-2),
    }


def trim(tt_tensor, target_shape):
    t = ttnn.to_torch(tt_tensor)
    slices = tuple(slice(0, s) for s in target_shape)
    return t[slices]


# ============================================================
# Helper: linear_relu_ln for standalone PT modules
# ============================================================

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


# ============================================================
# 1. MultiheadAttention
# ============================================================

def test_mha(device):
    from model.multihead_attention import MultiheadAttention

    results = {}

    # --- Self-attention ---
    torch.manual_seed(42)
    embed_dims, num_heads, bs, N = 512, 8, 1, 900

    attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=False)
    query = torch.randn(bs, N, embed_dims)
    with torch.no_grad():
        q_t = query.transpose(0, 1)
        pt_out = attn(q_t, q_t, q_t)[0].transpose(0, 1)
        pt_out = query + pt_out  # residual

    in_w = attn.in_proj_weight.data.clone()
    wq, wk, wv = in_w.chunk(3, dim=0)
    in_b = attn.in_proj_bias.data.clone()
    bq, bk, bv = in_b.chunk(3, dim=0)
    params = {
        "w_q": wq.t(), "w_k": wk.t(), "w_v": wv.t(),
        "b_q": bq, "b_k": bk, "b_v": bv,
        "w_out": attn.out_proj.weight.data.clone().t(),
        "b_out": attn.out_proj.bias.data.clone(),
    }
    tt_mha = MultiheadAttention(device, params, embed_dims, num_heads)
    q_tt = ttnn.from_torch(query.float(), layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_mha.run(q_tt, q_tt, q_tt, bs=bs, num_queries=N, num_keys=N)
    results["MHA self-attn"] = quick_compare(pt_out, trim(tt_out, pt_out.shape))

    # --- Cross-attention ---
    torch.manual_seed(123)
    M = 600
    query2 = torch.randn(bs, N, embed_dims)
    key2 = torch.randn(bs, M, embed_dims)
    value2 = torch.randn(bs, M, embed_dims)
    with torch.no_grad():
        pt_out2 = attn(query2.transpose(0, 1), key2.transpose(0, 1), value2.transpose(0, 1))[0].transpose(0, 1)
        pt_out2 = query2 + pt_out2

    q2_tt = ttnn.from_torch(query2.float(), layout=ttnn.TILE_LAYOUT, device=device)
    k2_tt = ttnn.from_torch(key2.float(), layout=ttnn.TILE_LAYOUT, device=device)
    v2_tt = ttnn.from_torch(value2.float(), layout=ttnn.TILE_LAYOUT, device=device)
    tt_out2 = tt_mha.run(q2_tt, k2_tt, v2_tt, bs=bs, num_queries=N, num_keys=M)
    results["MHA cross-attn"] = quick_compare(pt_out2, trim(tt_out2, pt_out2.shape))

    return results


# ============================================================
# 2. AsymmetricFFN
# ============================================================

def test_ffn(device):
    from model.asymmetric_ffn import AsymmetricFFN

    results = {}

    # --- Asymmetric 512→256 ---
    torch.manual_seed(42)
    bs, N, in_ch, out_ch, ff_ch = 1, 900, 512, 256, 1024

    pre_norm = nn.LayerNorm(in_ch)
    layers = nn.Sequential(
        nn.Sequential(nn.Linear(in_ch, ff_ch), nn.ReLU(inplace=True), nn.Dropout(0.0)),
        nn.Linear(ff_ch, out_ch), nn.Dropout(0.0),
    )
    identity_fc = nn.Linear(in_ch, out_ch)

    x = torch.randn(bs, N, in_ch)
    with torch.no_grad():
        x_normed = pre_norm(x)  # mmcv overwrites x in forward
        pt_out = identity_fc(x_normed) + layers(x_normed)

    params = {
        "pre_norm_weight": pre_norm.weight.data.clone(),
        "pre_norm_bias": pre_norm.bias.data.clone(),
        "fc1_weight": layers[0][0].weight.data.clone().t(),
        "fc1_bias": layers[0][0].bias.data.clone(),
        "fc2_weight": layers[1].weight.data.clone().t(),
        "fc2_bias": layers[1].bias.data.clone(),
        "identity_fc_weight": identity_fc.weight.data.clone().t(),
        "identity_fc_bias": identity_fc.bias.data.clone(),
    }
    tt_ffn = AsymmetricFFN(device, params, in_ch, out_ch, ff_ch)
    x_tt = ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_ffn.run(x_tt, bs=bs, num_tokens=N)
    results["FFN asymmetric (512→256)"] = quick_compare(pt_out, trim(tt_out, pt_out.shape))

    # --- Symmetric 256→256 ---
    torch.manual_seed(123)
    in_ch2 = 256
    pre_norm2 = nn.LayerNorm(in_ch2)
    layers2 = nn.Sequential(
        nn.Sequential(nn.Linear(in_ch2, ff_ch), nn.ReLU(inplace=True), nn.Dropout(0.0)),
        nn.Linear(ff_ch, out_ch), nn.Dropout(0.0),
    )
    x2 = torch.randn(bs, N, in_ch2)
    with torch.no_grad():
        x2_normed = pre_norm2(x2)
        pt_out2 = x2_normed + layers2(x2_normed)

    params2 = {
        "pre_norm_weight": pre_norm2.weight.data.clone(),
        "pre_norm_bias": pre_norm2.bias.data.clone(),
        "fc1_weight": layers2[0][0].weight.data.clone().t(),
        "fc1_bias": layers2[0][0].bias.data.clone(),
        "fc2_weight": layers2[1].weight.data.clone().t(),
        "fc2_bias": layers2[1].bias.data.clone(),
    }
    tt_ffn2 = AsymmetricFFN(device, params2, in_ch2, out_ch, ff_ch)
    x2_tt = ttnn.from_torch(x2.float(), layout=ttnn.TILE_LAYOUT, device=device)
    tt_out2 = tt_ffn2.run(x2_tt, bs=bs, num_tokens=N)
    results["FFN symmetric (256→256)"] = quick_compare(pt_out2, trim(tt_out2, pt_out2.shape))

    return results


# ============================================================
# 3. SparseBox3DEncoder
# ============================================================

def test_encoder(device):
    from model.sparse_box3d_encoder import SparseBox3DEncoder, _extract_linear_relu_ln_params

    results = {}

    X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = range(11)

    class _PTEncoder(nn.Module):
        def __init__(self, embed_dims, vel_dims=3, mode="cat", output_fc=False, in_loops=1, out_loops=4):
            super().__init__()
            self.mode = mode
            self.vel_dims = vel_dims
            def emb(inp, out): return nn.Sequential(*_linear_relu_ln(out, in_loops, out_loops, inp))
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

    def extract(enc):
        p = {}
        for name in ["pos_fc", "size_fc", "yaw_fc"]:
            p[name] = _extract_linear_relu_ln_params(list(getattr(enc, name).children()))
        if hasattr(enc, 'vel_fc'): p["vel_fc"] = _extract_linear_relu_ln_params(list(enc.vel_fc.children()))
        if enc.output_fc is not None: p["output_fc"] = _extract_linear_relu_ln_params(list(enc.output_fc.children()))
        return p

    # --- Cat mode ---
    torch.manual_seed(42)
    bs, N = 1, 900
    dims = [128, 32, 32, 64]
    pt_enc = _PTEncoder(dims, mode="cat", output_fc=False, in_loops=1, out_loops=4).eval()
    box = torch.randn(bs, N, 11)
    with torch.no_grad(): pt_out = pt_enc(box)

    tt_enc = SparseBox3DEncoder(device, extract(pt_enc), embed_dims=dims, mode="cat", has_output_fc=False, in_loops=1, out_loops=4)
    box_tt = ttnn.from_torch(box.float(), layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_enc.run(box_tt, bs=bs, num_anchor=N)
    results["Encoder cat mode"] = quick_compare(pt_out, trim(tt_out, pt_out.shape))

    # --- Add mode ---
    torch.manual_seed(123)
    pt_enc2 = _PTEncoder(256, mode="add", output_fc=True, in_loops=1, out_loops=2).eval()
    box2 = torch.randn(bs, N, 11)
    with torch.no_grad(): pt_out2 = pt_enc2(box2)

    tt_enc2 = SparseBox3DEncoder(device, extract(pt_enc2), embed_dims=256, mode="add", has_output_fc=True, in_loops=1, out_loops=2)
    box2_tt = ttnn.from_torch(box2.float(), layout=ttnn.TILE_LAYOUT, device=device)
    tt_out2 = tt_enc2.run(box2_tt, bs=bs, num_anchor=N)
    results["Encoder add mode"] = quick_compare(pt_out2, trim(tt_out2, pt_out2.shape))

    return results


# ============================================================
# 4. SparseBox3DRefinementModule
# ============================================================

def test_refinement(device):
    from model.refinement_module import SparseBox3DRefinementModule, _extract_linear_relu_ln_params

    results = {}

    X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = range(11)

    class _Scale(nn.Module):
        def __init__(self, s):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        def forward(self, x): return x * self.scale

    class _PTRefine(nn.Module):
        def __init__(self, embed_dims=256, output_dim=11, num_cls=10, refine_yaw=True, with_quality=True):
            super().__init__()
            self.output_dim, self.refine_yaw, self.with_quality = output_dim, refine_yaw, with_quality
            self.refine_state = [X, Y, Z, W, L, H] + ([SIN_YAW, COS_YAW] if refine_yaw else [])
            self.layers = nn.Sequential(*_linear_relu_ln(embed_dims, 2, 2), nn.Linear(embed_dims, output_dim), _Scale([1.0]*output_dim))
            self.cls_layers = nn.Sequential(*_linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, num_cls))
            if with_quality: self.quality_layers = nn.Sequential(*_linear_relu_ln(embed_dims, 1, 2), nn.Linear(embed_dims, 2))

        def forward(self, inst, anch, embed, ti=1.0):
            feat = inst + embed
            out = self.layers(feat)
            out[..., self.refine_state] = out[..., self.refine_state] + anch[..., self.refine_state]
            if self.output_dim > 8:
                if not isinstance(ti, torch.Tensor): ti = inst.new_tensor(ti)
                tr = torch.transpose(out[..., VX:], 0, -1)
                out[..., VX:] = torch.transpose(tr / ti, 0, -1) + anch[..., VX:]
            cls = self.cls_layers(inst)
            qt = self.quality_layers(feat) if self.with_quality else None
            return out, cls, qt

    def extract_ref(m):
        p = {}
        mods = list(m.layers.children())
        p["refine_layers"] = _extract_linear_relu_ln_params(mods[:-2])
        p["refine_final_weight"] = mods[-2].weight.data.clone().t()
        p["refine_final_bias"] = mods[-2].bias.data.clone()
        p["refine_scale"] = mods[-1].scale.data.clone()
        cls_mods = list(m.cls_layers.children())
        p["cls_layers"] = _extract_linear_relu_ln_params(cls_mods[:-1])
        p["cls_final_weight"] = cls_mods[-1].weight.data.clone().t()
        p["cls_final_bias"] = cls_mods[-1].bias.data.clone()
        if hasattr(m, 'quality_layers'):
            qt_mods = list(m.quality_layers.children())
            p["quality_layers"] = _extract_linear_relu_ln_params(qt_mods[:-1])
            p["quality_final_weight"] = qt_mods[-1].weight.data.clone().t()
            p["quality_final_bias"] = qt_mods[-1].bias.data.clone()
        return p

    torch.manual_seed(42)
    bs, N, E = 1, 900, 256
    pt_ref = _PTRefine(E, 11, 10, True, True).eval()
    inst = torch.randn(bs, N, E)
    anch = torch.randn(bs, N, 11)
    embed = torch.randn(bs, N, E)
    ti = torch.tensor([0.5] * bs)

    with torch.no_grad(): pt_out, pt_cls, pt_qt = pt_ref(inst, anch, embed, ti)

    tt_ref = SparseBox3DRefinementModule(device, extract_ref(pt_ref), E, 11, 10, True, True)
    inst_tt = ttnn.from_torch(inst.float(), layout=ttnn.TILE_LAYOUT, device=device)
    anch_tt = ttnn.from_torch(anch.float(), layout=ttnn.TILE_LAYOUT, device=device)
    embed_tt = ttnn.from_torch(embed.float(), layout=ttnn.TILE_LAYOUT, device=device)
    ti_tt = ttnn.from_torch(ti.reshape(1, 1, 1, bs).float(), layout=ttnn.TILE_LAYOUT, device=device)
    ti_tt = ttnn.reshape(ti_tt, (bs,))

    tt_out, tt_cls, tt_qt = tt_ref.run(inst_tt, anch_tt, embed_tt, ti_tt, bs=bs, num_anchor=N)

    results["Refinement anchor"] = quick_compare(pt_out, trim(tt_out, pt_out.shape))
    results["Refinement cls"] = quick_compare(pt_cls, trim(tt_cls, pt_cls.shape))
    results["Refinement quality"] = quick_compare(pt_qt, trim(tt_qt, pt_qt.shape))

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  Sparse4D Module PCC Summary: All Modules PyTorch vs TT-NN")
    print("=" * 70)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    all_results = {}

    try:
        compute_grid = device.compute_with_storage_grid_size()
        print(f"  Device: {device.arch()}, grid: {compute_grid.x}x{compute_grid.y}\n")

        tests = [
            ("MultiheadAttention", test_mha),
            ("AsymmetricFFN", test_ffn),
            ("SparseBox3DEncoder", test_encoder),
            ("RefinementModule", test_refinement),
        ]

        for group_name, test_fn in tests:
            print(f"  Running {group_name}...", end=" ", flush=True)
            t0 = time.time()
            results = test_fn(device)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")
            all_results.update(results)

        # Print summary table
        print(f"\n{'=' * 70}")
        print(f"  {'Module':<30} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>10} {'Pass':>6}")
        print(f"  {'-'*29:<30} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10} {'-'*5:>6}")

        all_pass = True
        for name, r in all_results.items():
            passed = r["pcc"] >= 0.999
            if not passed:
                all_pass = False
            mark = "OK" if passed else "FAIL"
            print(
                f"  {name:<30} {r['pcc']:>10.6f} {r['max_diff']:>10.4f} "
                f"{r['mean_diff']:>10.6f} {mark:>6}"
            )

        print(f"  {'-'*29:<30} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10} {'-'*5:>6}")
        avg_pcc = sum(r["pcc"] for r in all_results.values()) / len(all_results)
        status = "ALL PASS" if all_pass else "SOME FAILED"
        print(f"  {'Average':<30} {avg_pcc:>10.6f} {'':>10} {'':>10} {status:>6}")
        print(f"{'=' * 70}")

        # Additional note about modules not tested here
        print(f"\n  Note: ResNet50, FPN, DFA tested separately via:")
        print(f"    python test/resnet_pcc.py")
        print(f"    python test/fpn_pcc.py")
        print(f"    python test/dfa_pcc.py")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  FAILED: {str(e)[:200]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
