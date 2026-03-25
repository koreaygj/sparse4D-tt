"""
ResNet50 Backbone PCC Test: tt_resnet_bottleneck.py (7x7 conv1)
Tests the backbone that actually works with 256x704 input.
"""

import os
import sys

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.expanduser("~/project/tt-metal"))
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torchvision
import ttnn

from model.resnet_bottleneck import create_tt_resnet_bottleneck

SPATIAL_SHAPES = [(64, 176), (32, 88), (16, 44), (8, 22)]


def compute_pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == 0 and b.norm() == 0 else 0.0
    return (torch.dot(a, b) / denom).item()


def pytorch_backbone(model, x):
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


def tt_to_nchw(tt_tensor, batch, h, w, c):
    t = ttnn.to_torch(tt_tensor).float()
    t = t.view(-1, c)[:batch * h * w, :]
    t = t.view(batch, h, w, c).permute(0, 3, 1, 2).contiguous()
    return t


def main():
    print("=" * 60)
    print("  tt_resnet_bottleneck PCC Test (256x704)")
    print("=" * 60)

    batch_size = 6  # 6 cameras
    input_h, input_w = 256, 704

    torch.manual_seed(42)
    torch_input = torch.randn(batch_size, 3, input_h, input_w)

    # PyTorch golden
    pt_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    ).eval()
    with torch.no_grad():
        pt_outputs = pytorch_backbone(pt_model, torch_input)

    print(f"  PyTorch outputs:")
    for i, name in enumerate(["C2", "C3", "C4", "C5"]):
        print(f"    {name}: {list(pt_outputs[i].shape)}, "
              f"mean={pt_outputs[i].mean():.4f}, std={pt_outputs[i].std():.4f}, "
              f"max={pt_outputs[i].abs().max():.4f}")

    # TT-NN
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        grid = device.compute_with_storage_grid_size()
        print(f"\n  Device: {device.arch()}, grid: {grid.x}x{grid.y}")

        backbone, _ = create_tt_resnet_bottleneck(
            torch_model=pt_model,
            device=device,
            batch_size=batch_size,
            input_height=input_h,
            input_width=input_w,
        )

        # Prepare input: NCHW → NHWC → flattened
        imgs_nhwc = torch_input.permute(0, 2, 3, 1).contiguous()
        imgs_flat = imgs_nhwc.reshape(1, 1, batch_size * input_h * input_w, 3)
        tt_input = ttnn.from_torch(
            imgs_flat.float(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )

        print(f"  Running TT-NN backbone...")
        tt_outputs = backbone(tt_input)
        ttnn.synchronize_device(device)

        # Compare
        stage_names = ["C2", "C3", "C4", "C5"]
        expected_hw = [
            (input_h // 4, input_w // 4),
            (input_h // 8, input_w // 8),
            (input_h // 16, input_w // 16),
            (input_h // 32, input_w // 32),
        ]
        expected_ch = [256, 512, 1024, 2048]

        print(f"\n  {'Stage':<6} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>12} {'TT_max':>10} {'PT_max':>10}")
        print(f"  {'-'*5:<6} {'-'*9:>10} {'-'*9:>10} {'-'*11:>12} {'-'*9:>10} {'-'*9:>10}")

        for i, (name, pt_feat, tt_feat) in enumerate(
            zip(stage_names, pt_outputs, tt_outputs)
        ):
            h, w = expected_hw[i]
            c = expected_ch[i]
            tt_torch = tt_to_nchw(tt_feat, batch_size, h, w, c)

            pcc = compute_pcc(pt_feat, tt_torch)
            diff = (pt_feat.float() - tt_torch.float()).abs()
            print(f"  {name:<6} {pcc:>10.6f} {diff.max().item():>10.4f} "
                  f"{diff.mean().item():>12.6f} {tt_torch.abs().max().item():>10.4f} "
                  f"{pt_feat.abs().max().item():>10.4f}")

        # ============================================================
        # FPN PCC: run TT-NN FPN on TT backbone outputs
        # ============================================================
        print(f"\n  === FPN PCC (backbone output → FPN) ===")

        from model.fpn import FPN, _FPNParameters

        # Build PyTorch FPN
        pt_fpn = nn.Sequential()  # simple FPN
        # Use mmdet-style FPN from ckpt
        fpn_pt_path = os.path.join(os.path.dirname(__file__), "..", "ckpt", "fpn.pt")
        if os.path.exists(fpn_pt_path):
            from models.experimental.uniad.reference.fpn import FPN as PyTorchFPN
            pt_fpn_model = PyTorchFPN(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256, num_outs=4, start_level=0,
                add_extra_convs="on_output", relu_before_extra_convs=True,
            )
            fpn_sd = torch.load(fpn_pt_path, map_location="cpu")
            pt_fpn_model.load_state_dict(fpn_sd, strict=True)
            pt_fpn_model.eval()

            # PyTorch FPN forward
            with torch.no_grad():
                pt_fpn_outputs = pt_fpn_model(pt_outputs)

            # Build TT-NN FPN from same weights
            from model.fpn import preprocess_fpn_parameters
            fpn_params = preprocess_fpn_parameters(pt_fpn_model)
            tt_fpn = FPN(
                device=device,
                parameters=fpn_params,
                batch_size=batch_size,
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                model_config={
                    "WEIGHTS_DTYPE": ttnn.bfloat16,
                    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
                    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
                },
                input_spatial_shapes=SPATIAL_SHAPES,
            )

            # Run TT FPN on TT backbone outputs
            print(f"  Running TT-NN FPN...")
            # Move backbone outputs to DRAM (keep original dtype, don't typecast!)
            # FPN expects bfloat8_b/bfloat16 input, NOT float32
            tt_fpn_inputs = []
            for i in range(len(tt_outputs)):
                f = tt_outputs[i]
                f_dram = ttnn.to_memory_config(f, ttnn.DRAM_MEMORY_CONFIG)
                tt_fpn_inputs.append(f_dram)
                tt_outputs[i] = None
            tt_fpn_outputs = tt_fpn.run(tt_fpn_inputs, device)
            tt_fpn_outputs = [
                ttnn.typecast(f, ttnn.float32) if f.dtype != ttnn.float32 else f
                for f in tt_fpn_outputs
            ]
            ttnn.synchronize_device(device)

            print(f"\n  {'Level':<6} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>12} {'TT_max':>10} {'PT_max':>10}")
            print(f"  {'-'*5:<6} {'-'*9:>10} {'-'*9:>10} {'-'*11:>12} {'-'*9:>10} {'-'*9:>10}")

            for i, (h, w) in enumerate(SPATIAL_SHAPES):
                pt_out = pt_fpn_outputs[i]  # [6, 256, H, W]
                tt_raw = ttnn.to_torch(tt_fpn_outputs[i]).float()
                tt_out = tt_raw.view(-1, 256)[:batch_size * h * w, :]
                tt_out = tt_out.view(batch_size, h, w, 256).permute(0, 3, 1, 2).contiguous()

                pcc = compute_pcc(pt_out, tt_out)
                diff = (pt_out.float() - tt_out.float()).abs()
                print(f"  P{i+2:<4} {pcc:>10.6f} {diff.max().item():>10.4f} "
                      f"{diff.mean().item():>12.6f} {tt_out.abs().max().item():>10.4f} "
                      f"{pt_out.abs().max().item():>10.4f}")
        else:
            print(f"  Skipped: ckpt/fpn.pt not found")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
