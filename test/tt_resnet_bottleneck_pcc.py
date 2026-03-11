"""
TTNN ResNet50 Bottleneck PCC Test: PyTorch vs TTNN

Compares backbone feature maps [c2, c3, c4, c5] between:
  - PyTorch ResNet50 (float32 golden reference)
  - TTNN ResNet50 Bottleneck (BN-folded, bfloat16)
"""

import sys
import os

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.expanduser("~/project/tt-metal"))
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision
import ttnn
from model.tt_resnet_bottleneck import create_tt_resnet_bottleneck


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson Correlation Coefficient (float64 for accuracy)."""
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == 0 and b.norm() == 0 else 0.0
    return (torch.dot(a, b) / denom).item()


def pytorch_backbone_forward(model, x):
    """Run PyTorch ResNet50 backbone, return [c2, c3, c4, c5] in NCHW."""
    model.eval()
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        c2 = model.layer1(x)
        c3 = model.layer2(c2)
        c4 = model.layer3(c3)
        c5 = model.layer4(c4)
    return [c2, c3, c4, c5]


def tt_output_to_nchw(tt_tensor, batch_size, height, width, channels):
    """Convert TT flattened [1, 1, N*H*W, C] to NCHW torch tensor."""
    t = ttnn.to_torch(tt_tensor)
    t = t.view(-1, channels)
    t = t[: batch_size * height * width, :]
    t = t.view(batch_size, height, width, channels)
    t = t.permute(0, 3, 1, 2).contiguous()
    return t


def channel_mean(tensor_nchw):
    """Compute per-channel spatial mean: [N, C, H, W] -> [N, C]."""
    return tensor_nchw.float().mean(dim=(-2, -1))


def run_test(batch_size, input_height, input_width):
    """Run PCC comparison for TTNN ResNet50 Bottleneck."""
    input_shape = (batch_size, 3, input_height, input_width)

    torch.manual_seed(42)
    torch_input = torch.rand(input_shape, dtype=torch.float32)

    # PyTorch golden reference
    torch_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    ).eval()
    pt_outputs = pytorch_backbone_forward(torch_model, torch_input)

    # TTNN ResNet50 Bottleneck
    device = ttnn.open_device(device_id=0, l1_small_size=4 * 8192)
    try:
        compute_grid = device.compute_with_storage_grid_size()
        print(f"  Device: {device.arch()}, grid: {compute_grid.x}x{compute_grid.y}")
        print(f"  Input: batch={batch_size}, H={input_height}, W={input_width}")

        # Create fresh model for weight preprocessing
        torch_model_fresh = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        ).eval()

        tt_model, params = create_tt_resnet_bottleneck(
            torch_model=torch_model_fresh,
            device=device,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            out_indices=(0, 1, 2, 3),
            style="pytorch",
        )

        # Prepare input: NCHW → NHWC → flattened (1, 1, N*H*W, C)
        tt_input = torch_input.permute(0, 2, 3, 1)  # (N, H, W, C)
        tt_input = tt_input.reshape(1, 1, batch_size * input_height * input_width, 3)
        tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        tt_outputs = tt_model(tt_input)

        # Compare outputs
        stage_names = ["C2", "C3", "C4", "C5"]
        # Compute expected output sizes
        # After conv1 (7x7 s2 p3): h/2, w/2
        # After maxpool (3x3 s2 p1): h/4, w/4
        h0, w0 = input_height // 4, input_width // 4
        expected_hw = [
            (h0, w0),           # C2: stride 4
            (h0 // 2, w0 // 2), # C3: stride 8
            (h0 // 4, w0 // 4), # C4: stride 16
            (h0 // 8, w0 // 8), # C5: stride 32
        ]
        expected_ch = [256, 512, 1024, 2048]

        print(f"\n  {'Stage':<6} {'Spatial PCC':>12} {'Channel PCC':>12} {'MaxDiff':>10} {'MeanDiff':>12}")
        print(f"  {'-'*5:<6} {'-'*11:>12} {'-'*11:>12} {'-'*9:>10} {'-'*11:>12}")

        all_ok = True
        for i, (name, pt_feat, tt_feat) in enumerate(zip(stage_names, pt_outputs, tt_outputs)):
            h, w = expected_hw[i]
            c = expected_ch[i]
            tt_torch = tt_output_to_nchw(tt_feat, batch_size, h, w, c)

            spatial_pcc = compute_pcc(pt_feat, tt_torch)
            pt_ch = channel_mean(pt_feat)
            tt_ch = channel_mean(tt_torch)
            ch_pcc = compute_pcc(pt_ch, tt_ch)

            abs_diff = (pt_feat.float() - tt_torch.float()).abs()
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()

            print(f"  {name:<6} {spatial_pcc:>12.6f} {ch_pcc:>12.6f} {max_diff:>10.4f} {mean_diff:>12.6f}")

            if spatial_pcc < 0.90:
                all_ok = False

        return all_ok

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED: {str(e)[:100]}")
        return False

    finally:
        ttnn.close_device(device)


def main():
    print("=" * 65)
    print("TTNN ResNet50 Bottleneck PCC Test: PyTorch vs TTNN")
    print("=" * 65)

    # Test with Sparse4D input dimensions
    configs = [
        {"batch": 6, "height": 256, "width": 704, "name": "Sparse4D (6×256×704)"},
    ]

    results = {}
    for cfg in configs:
        print(f"\n{'=' * 65}")
        print(f"  Config: {cfg['name']}")
        print(f"{'=' * 65}")
        ok = run_test(cfg["batch"], cfg["height"], cfg["width"])
        results[cfg["name"]] = ok

    print(f"\n{'=' * 65}")
    print("Summary:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
