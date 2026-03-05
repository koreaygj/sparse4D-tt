"""
ResNet50 PCC Test: PyTorch vs TT-NN
Compares backbone feature maps with different precision configs.
"""

import sys
import os

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.expanduser("~/project/tt-metal"))
sys.path.insert(0, TT_METAL_HOME)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.vision.classification.resnet50.ttnn_resnet.tt.custom_preprocessing import (
    create_custom_mesh_preprocessor,
)
from model.resnet import resnet50


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if a.norm() == 0 and b.norm() == 0 else 0.0
    return (torch.dot(a, b) / denom).item()


def pytorch_backbone_forward(model, x):
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
    t = ttnn.to_torch(tt_tensor)
    t = t.view(-1, channels)
    t = t[: batch_size * height * width, :]
    t = t.view(batch_size, height, width, channels)
    t = t.permute(0, 3, 1, 2).contiguous()
    return t


def run_single_config(batch_size, model_config, config_name):
    """Run a single config in isolation (fresh device per config)."""
    input_shape = (batch_size, 3, 224, 224)

    torch.manual_seed(42)
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

    # PyTorch golden
    torch_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    ).to(torch.bfloat16).eval()
    pt_outputs = pytorch_backbone_forward(torch_model, torch_input)

    # TT-NN
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        compute_grid = device.compute_with_storage_grid_size()
        print(f"  Device: {device.arch()}, grid: {compute_grid.x}x{compute_grid.y}, batch={batch_size}")

        torch_model_f32 = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        ).eval()

        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model_f32,
            custom_preprocessor=create_custom_mesh_preprocessor(),
            device=None,
        )

        tt_model = resnet50(
            device=device,
            parameters=parameters,
            batch_size=batch_size,
            model_config=model_config,
            input_shape=input_shape,
            kernel_size=3,
            stride=2,
            dealloc_input=True,
            final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_model.conv1_config.enable_activation_reuse = False

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_input = ttnn.to_device(tt_input, device)
        tt_outputs = tt_model(tt_input, device, {})

        stage_names = ["C2", "C3", "C4", "C5"]
        expected_hw = [(56, 56), (28, 28), (14, 14), (7, 7)]
        expected_ch = [256, 512, 1024, 2048]

        print(f"  {'Stage':<6} {'PCC':>10} {'MaxDiff':>10} {'MeanDiff':>12}")
        print(f"  {'-'*5:<6} {'-'*9:>10} {'-'*9:>10} {'-'*11:>12}")

        for i, (name, pt_feat, tt_feat) in enumerate(
            zip(stage_names, pt_outputs, tt_outputs)
        ):
            h, w = expected_hw[i]
            c = expected_ch[i]
            tt_torch = tt_output_to_nchw(tt_feat, batch_size, h, w, c)
            # Slice PyTorch output to match batch_size
            pcc = compute_pcc(pt_feat, tt_torch)
            abs_diff = (pt_feat.float() - tt_torch.float()).abs()
            max_diff = abs_diff.max().item()
            mean_diff = abs_diff.mean().item()
            print(f"  {name:<6} {pcc:>10.6f} {max_diff:>10.4f} {mean_diff:>12.6f}")

        return True

    except Exception as e:
        err_msg = str(e).split("\n")[0][:100]
        print(f"  FAILED: {err_msg}")
        return False

    finally:
        ttnn.close_device(device)


def main():
    print("=" * 60)
    print("ResNet50 PCC Test: PyTorch vs TT-NN")
    print("=" * 60)

    configs = [
        # batch=16, low precision (fits N300 L1)
        {
            "batch": 16,
            "name": "batch=16, bfloat8_b + LoFi",
            "config": {
                "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
                "WEIGHTS_DTYPE": ttnn.bfloat8_b,
                "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
            },
        },
        # batch=8, high precision (should fit N300 L1)
        {
            "batch": 8,
            "name": "batch=8, bfloat16 + HiFi2",
            "config": {
                "MATH_FIDELITY": ttnn.MathFidelity.HiFi2,
                "WEIGHTS_DTYPE": ttnn.bfloat16,
                "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            },
        },
    ]

    results = {}
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"  {cfg['name']}")
        print(f"{'='*60}")
        ok = run_single_config(cfg["batch"], cfg["config"], cfg["name"])
        results[cfg["name"]] = ok

    print(f"\n{'='*60}")
    print("Summary:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
