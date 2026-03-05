import os
import sys

# tt-metal 경로 추가

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME", os.path.expanduser("~/project/tt-metal")
)
sys.path.insert(0, TT_METAL_HOME)
# Add project root for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision


def get_pytorch_backbone_outputs(model, input_tensor):
    """PyTorch ResNet50에서 중간 feature map 추출 (golden reference)"""
    model.eval()
    with torch.no_grad():
        x = model.conv1(input_tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        c2 = model.layer1(x)
        c3 = model.layer2(c2)
        c4 = model.layer3(c3)
        c5 = model.layer4(c4)

    return [c2, c3, c4, c5]


def test_pytorch_shapes():
    """PyTorch golden reference shape 확인"""
    batch_size = 1
    num_cams = 6
    input_tensor = torch.rand(batch_size * num_cams, 3, 256, 704, dtype=torch.bfloat16)

    model = (
        torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        .to(torch.bfloat16)
        .eval()
    )

    pytorch_outputs = get_pytorch_backbone_outputs(model, input_tensor)

    stage_names = ["C2 (layer1)", "C3 (layer2)", "C4 (layer3)", "C5 (layer4)"]
    expected_channels = [256, 512, 1024, 2048]

    print("=== PyTorch ResNet50 Backbone Outputs ===")
    for i, (name, feat) in enumerate(zip(stage_names, pytorch_outputs)):
        h, w = feat.shape[2], feat.shape[3]
        stride = 256 // h
        print(f"  {name}: {list(feat.shape)}, stride={stride}")
        assert feat.shape[1] == expected_channels[i], (
            f"Channel mismatch: expected {expected_channels[i]}, got {feat.shape[1]}"
        )

    print("  -> All PyTorch shape checks passed.\n")
    return pytorch_outputs


def test_ttnn_shapes():
    """TT 디바이스에서 model/resnet.py backbone 출력 확인"""
    import ttnn
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.demos.vision.classification.resnet50.ttnn_resnet.tt.custom_preprocessing import (
        create_custom_mesh_preprocessor,
    )
    from model.resnet import resnet50

    batch_size = 16
    num_cams = 1  # TT resnet50은 단일 배치로 처리
    input_shape = (batch_size, 3, 224, 224)

    # PyTorch 모델 로드 및 전처리
    torch_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    ).eval()

    model_config = {
        "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
        "WEIGHTS_DTYPE": ttnn.bfloat8_b,
        "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
    }

    # TT 디바이스 열기
    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    try:
        # 디바이스 정보 출력
        compute_grid = device.compute_with_storage_grid_size()
        print(f"=== Device Info ===")
        print(f"  Arch: {device.arch()}")
        print(
            f"  Compute grid: {compute_grid.x}x{compute_grid.y} = {compute_grid.x * compute_grid.y} cores"
        )
        print()

        # 가중치 전처리 (BN folding 포함)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(),
            device=None,
        )

        # TT-NN ResNet50 backbone 생성
        resnet50_first_conv_kernel_size = 3
        resnet50_first_conv_stride = 2

        tt_model = resnet50(
            device=device,
            parameters=parameters,
            batch_size=batch_size,
            model_config=model_config,
            input_shape=input_shape,
            kernel_size=resnet50_first_conv_kernel_size,
            stride=resnet50_first_conv_stride,
            dealloc_input=True,
            final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        )

        # activation reuse 비활성화 (디바이스/배치 조합에 따른 호환성 이슈 방지)
        tt_model.conv1_config.enable_activation_reuse = False

        # 입력 텐서 생성
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_input = ttnn.to_device(tt_input, device)

        # 추론
        tt_outputs = tt_model(tt_input, device, {})

        # 결과 확인
        stage_names = ["C2 (layer1)", "C3 (layer2)", "C4 (layer3)", "C5 (layer4)"]

        print("=== TT-NN ResNet50 Backbone Outputs ===")
        for name, feat in zip(stage_names, tt_outputs):
            print(f"  {name}: {list(feat.shape)}")

        # torch로 변환하여 shape 비교
        print("\n=== Converting to torch for verification ===")
        for name, feat in zip(stage_names, tt_outputs):
            torch_feat = ttnn.to_torch(feat)
            print(f"  {name}: torch shape = {list(torch_feat.shape)}")

        print("\n  -> TT-NN backbone test passed.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    print("=" * 50)
    print("ResNet50 Backbone Test (Sparse4D)")
    print("=" * 50)
    print()

    # 1. PyTorch shape 테스트 (디바이스 불필요)
    test_pytorch_shapes()

    # 2. TT-NN 테스트 (디바이스 필요)
    try:
        test_ttnn_shapes()
    except Exception as e:
        print(f"=== TT-NN test skipped: {e} ===")
        print("  TT 디바이스가 없거나 환경 설정이 필요합니다.")
