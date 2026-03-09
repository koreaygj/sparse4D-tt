"""
FPN PCC Test: PyTorch vs TT-NN
Compares FPN outputs using pretrained weights from ckpt/fpn.pt.
"""

import os
import sys

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.expanduser("~/project/tt-metal")
)
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import ttnn

# Reference PyTorch FPN from tt-metal
from models.experimental.uniad.reference.fpn import FPN as PyTorchFPN

from model.fpn import FPN as TtFPN, preprocess_fpn_parameters


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == 0 and b.norm() == 0 else 0.0
    return (torch.dot(a, b) / denom).item()


def tt_output_to_nchw(tt_tensor, batch_size, height, width, channels):
    t = ttnn.to_torch(tt_tensor)
    t = t.view(-1, channels)
    t = t[: batch_size * height * width, :]
    t = t.view(batch_size, height, width, channels)
    t = t.permute(0, 3, 1, 2).contiguous()
    return t


def create_pytorch_fpn():
    """Create PyTorch FPN matching Sparse4D config and load ckpt/fpn.pt weights."""
    model = PyTorchFPN(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
    )
    state_dict = torch.load(
        os.path.join(os.path.dirname(__file__), "..", "ckpt", "fpn.pt"),
        map_location="cpu",
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_fpn_pcc(batch_size, model_config, config_name):
    """Run FPN PCC test: PyTorch golden vs TT-NN."""
    # Sparse4D spatial shapes for ResNet50 with 256x704 input
    # Using standard 224x224 backbone shapes for simplicity
    input_spatial_shapes = [(56, 56), (28, 28), (14, 14), (7, 7)]
    in_channels = [256, 512, 1024, 2048]
    out_channels = 256

    torch.manual_seed(42)

    # Create fake backbone outputs (NCHW)
    pt_inputs = []
    for i, (h, w) in enumerate(input_spatial_shapes):
        pt_inputs.append(torch.randn(batch_size, in_channels[i], h, w))

    # PyTorch golden
    pt_model = create_pytorch_fpn()
    with torch.no_grad():
        pt_outputs = pt_model(pt_inputs)

    # TT-NN
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        compute_grid = device.compute_with_storage_grid_size()
        print(
            f"  Device: {device.arch()}, grid: {compute_grid.x}x{compute_grid.y}, batch={batch_size}"
        )

        # Preprocess parameters from PyTorch model
        parameters = preprocess_fpn_parameters(pt_model)

        # Create TT FPN
        tt_model = TtFPN(
            device=device,
            parameters=parameters,
            batch_size=batch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            model_config=model_config,
            input_spatial_shapes=input_spatial_shapes,
        )

        # Convert inputs: NCHW -> flattened [1, 1, N*H*W, C]
        tt_inputs = []
        for i, pt_feat in enumerate(pt_inputs):
            h, w = input_spatial_shapes[i]
            # NCHW -> NHWC -> flattened
            x = pt_feat.permute(0, 2, 3, 1).contiguous()
            x = x.reshape(1, 1, batch_size * h * w, in_channels[i])
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.to_device(x, device)
            tt_inputs.append(x)

        # Run TT FPN
        tt_outputs = tt_model.run(tt_inputs, device)

        # Compare outputs
        level_names = ["P2", "P3", "P4", "P5"]
        print(f"  {'Level':<6} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>12}")
        print(f"  {'-' * 5:<6} {'-' * 9:>10} {'-' * 9:>10} {'-' * 11:>12}")

        for i, (name, pt_out, tt_out) in enumerate(
            zip(level_names, pt_outputs, tt_outputs)
        ):
            h, w = input_spatial_shapes[i]
            tt_torch = tt_output_to_nchw(tt_out, batch_size, h, w, out_channels)
            pcc = compute_pcc(pt_out, tt_torch)
            abs_diff = (pt_out.float() - tt_torch.float()).abs()
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()
            print(f"  {name:<6} {pcc:>10.6f} {max_diff:>10.4f} {mean_diff:>12.6f}")

        return True

    except Exception as e:
        import traceback

        traceback.print_exc()
        err_msg = str(e).split("\n")[0][:100]
        print(f"  FAILED: {err_msg}")
        return False

    finally:
        ttnn.close_device(device)


def main():
    print("=" * 60)
    print("FPN PCC Test: PyTorch vs TT-NN")
    print("=" * 60)

    configs = [
        {
            "batch": 1,
            "name": "batch=1, bfloat16 + HiFi2",
            "config": {
                "MATH_FIDELITY": ttnn.MathFidelity.HiFi2,
                "WEIGHTS_DTYPE": ttnn.bfloat16,
                "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            },
        },
        {
            "batch": 8,
            "name": "batch=8, bfloat8_b + LoFi",
            "config": {
                "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
                "WEIGHTS_DTYPE": ttnn.bfloat8_b,
                "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
            },
        },
    ]

    results = {}
    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"  {cfg['name']}")
        print(f"{'=' * 60}")
        ok = run_fpn_pcc(cfg["batch"], cfg["config"], cfg["name"])
        results[cfg["name"]] = ok

    print(f"\n{'=' * 60}")
    print(f"\n{'='*60}")
    print("Summary:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
