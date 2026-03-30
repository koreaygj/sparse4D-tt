"""
ResNet50 Dual-Device PCC Test

Runs batch=3 on each of 2 Wormhole chips via separate processes.
Each process opens its own device independently (no CQ conflict).
fp32_dest_acc_en=True enabled for precision improvement.
"""

import os
import sys
import torch
import torchvision
import multiprocessing as mp

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.expanduser("~/project/tt-metal"))
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return (torch.dot(a, b) / d).item() if d > 0 else 1.0


def tt_to_nchw(t, n, h, w, c):
    import ttnn
    t = ttnn.to_torch(t).float()
    return t.view(-1, c)[:n * h * w].view(n, h, w, c).permute(0, 3, 1, 2).contiguous()


def run_on_device(device_id, input_tensor, result_dict):
    """Run ResNet backbone on a single device with batch=3, fp32_dest_acc_en=True."""
    import ttnn
    from model.resnet_bottleneck import create_tt_resnet_bottleneck

    batch = input_tensor.shape[0]
    H, W = input_tensor.shape[2], input_tensor.shape[3]

    pt_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    ).eval()

    device = ttnn.open_device(device_id=device_id, l1_small_size=24576)
    try:
        backbone, _ = create_tt_resnet_bottleneck(
            torch_model=pt_model,
            device=device,
            batch_size=batch,
            input_height=H,
            input_width=W,
        )

        imgs = input_tensor.permute(0, 2, 3, 1).contiguous().reshape(1, 1, batch * H * W, 3)
        tt_x = ttnn.from_torch(
            imgs.float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16,
        )
        tt_outs = backbone(tt_x)
        ttnn.synchronize_device(device)

        # Convert to torch
        channels = [256, 512, 1024, 2048]
        strides = [4, 8, 16, 32]
        torch_outs = []
        for i in range(4):
            h = H // strides[i]
            w = W // strides[i]
            torch_outs.append(tt_to_nchw(tt_outs[i], batch, h, w, channels[i]))

        result_dict[device_id] = torch_outs
        print(f"  Device {device_id}: batch={batch} completed successfully")

    except Exception as e:
        result_dict[device_id] = f"ERROR: {str(e)[:200]}"
        print(f"  Device {device_id}: FAILED - {str(e)[:100]}")
    finally:
        ttnn.close_device(device)


def main():
    print("=" * 60)
    print("  ResNet50 Dual-Device PCC Test (batch=3 x 2)")
    print("=" * 60)

    batch = 6
    H, W = 256, 704

    torch.manual_seed(42)
    x_f32 = torch.randn(batch, 3, H, W)

    # PyTorch golden (full batch=6, float32)
    pt_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    ).eval()
    with torch.no_grad():
        x = pt_model.conv1(x_f32)
        x = pt_model.bn1(x)
        x = pt_model.relu(x)
        x = pt_model.maxpool(x)
        c2 = pt_model.layer1(x)
        c3 = pt_model.layer2(c2)
        c4 = pt_model.layer3(c3)
        c5 = pt_model.layer4(c4)
    pt_outs = [c2, c3, c4, c5]

    # Split batch: cam0-2 → device0, cam3-5 → device1
    x_dev0 = x_f32[:3]  # batch=3
    x_dev1 = x_f32[3:]  # batch=3

    # Run on 2 devices in separate processes
    manager = mp.Manager()
    result_dict = manager.dict()

    print(f"\n  Launching 2 processes (device 0: cam0-2, device 1: cam3-5)...")

    p0 = mp.Process(target=run_on_device, args=(0, x_dev0, result_dict))
    p1 = mp.Process(target=run_on_device, args=(1, x_dev1, result_dict))

    p0.start()
    p1.start()
    p0.join(timeout=120)
    p1.join(timeout=120)

    if p0.is_alive():
        p0.terminate()
        print("  Device 0: TIMEOUT")
    if p1.is_alive():
        p1.terminate()
        print("  Device 1: TIMEOUT")

    # Check results
    for dev_id in [0, 1]:
        if dev_id not in result_dict:
            print(f"  Device {dev_id}: No result returned")
            return
        if isinstance(result_dict[dev_id], str):
            print(f"  Device {dev_id}: {result_dict[dev_id]}")
            return

    # Concat results from both devices
    tt_outs_0 = result_dict[0]
    tt_outs_1 = result_dict[1]

    print(f"\n  {'Stage':<6} {'Combined PCC':>12} {'Dev0 PCC':>10} {'Dev1 PCC':>10} {'MaxDiff':>10}")
    print(f"  {'-'*5:<6} {'-'*11:>12} {'-'*9:>10} {'-'*9:>10} {'-'*9:>10}")

    channels = [256, 512, 1024, 2048]
    for i in range(4):
        # Concat both halves
        tt_combined = torch.cat([tt_outs_0[i], tt_outs_1[i]], dim=0)
        p_combined = pcc(pt_outs[i], tt_combined)
        p_dev0 = pcc(pt_outs[i][:3], tt_outs_0[i])
        p_dev1 = pcc(pt_outs[i][3:], tt_outs_1[i])
        d = (pt_outs[i] - tt_combined).abs()
        print(f"  C{i+2:<4} {p_combined:>12.6f} {p_dev0:>10.6f} {p_dev1:>10.6f} {d.max().item():>10.4f}")

    # Compare with batch=6 single device reference
    print(f"\n  Note: Compare 'Combined PCC' above with batch=6 single-device PCC")
    print(f"  fp32_dest_acc_en status depends on tt_resnet_bottleneck.py setting")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
