# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# TTNN ResNet50 Bottleneck Backbone for Sparse4D
#
# Standard ResNet50 with Bottleneck blocks, implemented in TTNN.
# BN is folded into conv weights at preprocessing time.
#
# Architecture (ResNet50):
#   Input (N, 3, H, W) → conv1 7x7 s2 → ReLU → maxpool 3x3 s2
#   → Layer1 (3 Bottleneck blocks, 64→256ch,  stride 1)
#   → Layer2 (4 Bottleneck blocks, 128→512ch, stride 2)
#   → Layer3 (6 Bottleneck blocks, 256→1024ch, stride 2)
#   → Layer4 (3 Bottleneck blocks, 512→2048ch, stride 2)
#   → Outputs multi-scale features [c2, c3, c4, c5] at strides [4, 8, 16, 32]
# =============================================================================

import torch
import torch.nn as nn
import ttnn
from typing import List, Tuple, Optional, Dict
from loguru import logger


# =============================================================================
# Weight Preprocessing: BN Folding + TTNN Parameter Conversion
# =============================================================================

def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold BatchNorm parameters into Conv2d weight and bias.

    Produces equivalent weight/bias such that:
        BN(Conv(x)) ≈ Conv_folded(x)

    Args:
        conv: Conv2d module
        bn: BatchNorm2d module (must have running_mean/running_var)

    Returns:
        (folded_weight, folded_bias) as float32 tensors
    """
    w = conv.weight.clone().float()
    b = conv.bias.clone().float() if conv.bias is not None else torch.zeros(conv.out_channels)

    mu = bn.running_mean.float()
    var = bn.running_var.float()
    gamma = bn.weight.float() if bn.affine else torch.ones_like(mu)
    beta = bn.bias.float() if bn.affine else torch.zeros_like(mu)
    eps = bn.eps

    # scale = gamma / sqrt(var + eps)
    scale = gamma / torch.sqrt(var + eps)

    # w_folded = w * scale[:, None, None, None]
    w_folded = w * scale.view(-1, 1, 1, 1)

    # b_folded = (b - mu) * scale + beta
    b_folded = (b - mu) * scale + beta

    return w_folded, b_folded


def preprocess_resnet50_parameters(
    model: nn.Module,
    device=None,
) -> dict:
    """Preprocess a PyTorch ResNet50 model into TTNN-ready parameters.

    Folds all BatchNorm layers into their preceding Conv2d layers.
    Converts weights to ttnn tensors in float32 format.

    Args:
        model: PyTorch ResNet50 model (torchvision-style with conv1/bn1/layer1-4)
        device: ttnn device (optional, if None weights stay on host)

    Returns:
        Dictionary with preprocessed parameters:
        {
            "conv1": {"weight": ttnn.Tensor, "bias": ttnn.Tensor},
            "layer1": {0: {"conv1": {...}, "conv2": {...}, "conv3": {...}, "downsample": {...}}, ...},
            "layer2": {...},
            "layer3": {...},
            "layer4": {...},
        }
    """
    model.eval()
    parameters = {}

    # Conv1 + BN1
    weight, bias = fold_bn_into_conv(model.conv1, model.bn1)
    parameters["conv1"] = {
        "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
        "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.float32),
    }

    # Layers 1-4
    for layer_idx in range(1, 5):
        layer = getattr(model, f"layer{layer_idx}")
        layer_params = {}

        for block_idx, block in enumerate(layer):
            block_params = {}

            # conv1 + bn1, conv2 + bn2, conv3 + bn3
            for conv_name, bn_name in [("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")]:
                conv = getattr(block, conv_name)
                bn = getattr(block, bn_name)
                w, b = fold_bn_into_conv(conv, bn)
                block_params[conv_name] = {
                    "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.float32),
                }

            # Downsample path (if present)
            if block.downsample is not None:
                ds_conv = block.downsample[0]
                ds_bn = block.downsample[1]
                w, b = fold_bn_into_conv(ds_conv, ds_bn)
                block_params["downsample"] = {
                    "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                    "bias": ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=ttnn.float32),
                }

            layer_params[block_idx] = block_params
        parameters[f"layer{layer_idx}"] = layer_params

    return parameters


def infer_conv_shapes(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> dict:
    """Run a forward pass to infer input shapes for each conv layer.

    Args:
        model: PyTorch ResNet50 model
        input_tensor: Sample input tensor (N, C, H, W)

    Returns:
        Dictionary mapping layer paths to (batch_size, input_height, input_width) tuples
    """
    shapes = {}
    hooks = []
    batch_size = input_tensor.shape[0]

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            shapes[name] = {
                "batch_size": inp.shape[0],
                "input_height": inp.shape[2],
                "input_width": inp.shape[3],
            }
        return hook_fn

    # Register hooks on all Conv2d layers
    hooks.append(model.conv1.register_forward_hook(make_hook("conv1")))

    for layer_idx in range(1, 5):
        layer = getattr(model, f"layer{layer_idx}")
        for block_idx, block in enumerate(layer):
            prefix = f"layer{layer_idx}.{block_idx}"
            hooks.append(block.conv1.register_forward_hook(make_hook(f"{prefix}.conv1")))
            hooks.append(block.conv2.register_forward_hook(make_hook(f"{prefix}.conv2")))
            hooks.append(block.conv3.register_forward_hook(make_hook(f"{prefix}.conv3")))
            if block.downsample is not None:
                hooks.append(block.downsample[0].register_forward_hook(make_hook(f"{prefix}.downsample")))

    model.eval()
    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    return shapes


# =============================================================================
# TTNN Conv2d Wrapper
# =============================================================================

class Conv2dOp:
    """TTNN conv2d wrapper with pre-loaded weights."""

    def __init__(
        self,
        params: dict,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        batch_size: int,
        input_height: int,
        input_width: int,
        groups: int = 1,
        activation=None,
        weights_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation: bool = False,
        act_block_h_override: int = 0,
        math_fidelity=ttnn.MathFidelity.LoFi,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.groups = groups

        self.weight = params["weight"]
        self.bias = params.get("bias", None)

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=deallocate_activation,
            enable_act_double_buffer=False,
            reshard_if_not_optimal=True,
            activation=activation,
        )
        if act_block_h_override > 0:
            self.conv_config.act_block_h_override = act_block_h_override

    def __call__(self, x: ttnn.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        [x, [out_h, out_w], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            batch_size=self.batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x, out_h, out_w


# =============================================================================
# Bottleneck Block
# =============================================================================

class Bottleneck:
    """ResNet Bottleneck block: conv1x1 → conv3x3 → conv1x1 + residual.

    PyTorch style: stride applied to conv2 (3x3).
    All BN is pre-folded into conv weights.
    """
    expansion = 4

    def __init__(
        self,
        params: dict,
        shapes: dict,
        device,
        planes: int,
        stride: int = 1,
        has_downsample: bool = False,
        style: str = "pytorch",
        activation_dtype=ttnn.bfloat16,
        conv3_block_sharded: bool = False,
        downsample_block_sharded: bool = False,
    ):
        self.device = device
        self.has_downsample = has_downsample
        self.activation_dtype = activation_dtype

        if style == "pytorch":
            conv1_stride = 1
            conv2_stride = stride
        else:  # caffe
            conv1_stride = stride
            conv2_stride = 1

        # conv1: 1x1 reduction
        s = shapes["conv1"]
        self.conv1 = Conv2dOp(
            params["conv1"], device,
            in_channels=params["conv1"]["weight"].shape[1],
            out_channels=params["conv1"]["weight"].shape[0],
            kernel_size=(1, 1),
            stride=(conv1_stride, conv1_stride),
            padding=(0, 0),
            batch_size=s["batch_size"],
            input_height=s["input_height"],
            input_width=s["input_width"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )

        # conv2: 3x3 spatial
        s = shapes["conv2"]
        self.conv2 = Conv2dOp(
            params["conv2"], device,
            in_channels=params["conv2"]["weight"].shape[1],
            out_channels=params["conv2"]["weight"].shape[0],
            kernel_size=(3, 3),
            stride=(conv2_stride, conv2_stride),
            padding=(1, 1),
            batch_size=s["batch_size"],
            input_height=s["input_height"],
            input_width=s["input_width"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            act_block_h_override=32,
        )

        # conv3: 1x1 expansion (no activation - added after residual)
        # Use BLOCK_SHARDED for deeper layers to avoid L1 overflow with large channels
        conv3_shard = (
            ttnn.TensorMemoryLayout.BLOCK_SHARDED if conv3_block_sharded
            else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        )
        s = shapes["conv3"]
        self.conv3 = Conv2dOp(
            params["conv3"], device,
            in_channels=params["conv3"]["weight"].shape[1],
            out_channels=params["conv3"]["weight"].shape[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=s["batch_size"],
            input_height=s["input_height"],
            input_width=s["input_width"],
            activation=None,
            deallocate_activation=True,
            shard_layout=conv3_shard,
        )

        # Downsample path
        # Use BLOCK_SHARDED for deeper layers (layer3/4) where output channels
        # are 1024/2048 and HEIGHT_SHARDED would exceed L1 per core
        if has_downsample:
            ds_shard = (
                ttnn.TensorMemoryLayout.BLOCK_SHARDED if downsample_block_sharded
                else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            )
            s = shapes["downsample"]
            self.downsample = Conv2dOp(
                params["downsample"], device,
                in_channels=params["downsample"]["weight"].shape[1],
                out_channels=params["downsample"]["weight"].shape[0],
                kernel_size=(1, 1),
                stride=(stride, stride),
                padding=(0, 0),
                batch_size=s["batch_size"],
                input_height=s["input_height"],
                input_width=s["input_width"],
                activation=None,
                activation_dtype=activation_dtype,
                shard_layout=ds_shard,
            )

    def __call__(self, x_identity: ttnn.Tensor) -> ttnn.Tensor:
        # Main path: conv1 → conv2 → conv3
        x, out_h, out_w = self.conv1(x_identity)

        if self.activation_dtype == ttnn.bfloat8_b:
            x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            x_identity = ttnn.add(x_identity, 0.0, dtype=ttnn.bfloat8_b)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x, _, _ = self.conv2(x)
        x, _, _ = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Residual path
        if self.has_downsample:
            x_identity, _, _ = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)

        # Add + ReLU
        x = ttnn.add(x, x_identity)
        x = ttnn.relu(x)

        ttnn.deallocate(x_identity)
        return x


# =============================================================================
# Residual Layer (sequence of Bottleneck blocks)
# =============================================================================

class ResLayer:
    """A sequence of Bottleneck blocks forming one residual layer.

    Args:
        conv3_block_sharded: Use BLOCK_SHARDED for conv3 (1x1 expansion).
            Needed for layer3/4 where output channels (1024/2048) are too large
            for HEIGHT_SHARDED to fit in L1.
        downsample_block_sharded: Use BLOCK_SHARDED for downsample conv.
            Same reason — large channel counts in deeper layers.
    """

    def __init__(
        self,
        params: dict,
        shapes: dict,
        device,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        style: str = "pytorch",
        activation_dtype=ttnn.bfloat16,
        conv3_block_sharded: bool = False,
        downsample_block_sharded: bool = False,
    ):
        expansion = Bottleneck.expansion
        has_downsample = (stride != 1 or inplanes != planes * expansion)

        self.blocks = []

        # First block (may have downsample + stride)
        block_shapes = {
            "conv1": shapes["conv1"],
            "conv2": shapes["conv2"],
            "conv3": shapes["conv3"],
        }
        if has_downsample:
            block_shapes["downsample"] = shapes["downsample"]

        self.blocks.append(
            Bottleneck(
                params[0], block_shapes, device,
                planes=planes,
                stride=stride,
                has_downsample=has_downsample,
                style=style,
                activation_dtype=activation_dtype,
                conv3_block_sharded=conv3_block_sharded,
                downsample_block_sharded=downsample_block_sharded,
            )
        )

        # Remaining blocks (no downsample, stride=1)
        for j in range(1, num_blocks):
            block_shapes_j = {}
            for conv_name in ["conv1", "conv2", "conv3"]:
                key = f"block{j}.{conv_name}"
                if key in shapes:
                    block_shapes_j[conv_name] = shapes[key]
                else:
                    block_shapes_j[conv_name] = shapes.get("conv1", shapes["conv1"])

            self.blocks.append(
                Bottleneck(
                    params[j], block_shapes_j, device,
                    planes=planes,
                    stride=1,
                    has_downsample=False,
                    style=style,
                    activation_dtype=activation_dtype,
                    conv3_block_sharded=conv3_block_sharded,
                    downsample_block_sharded=False,
                )
            )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# =============================================================================
# Full ResNet50 Backbone
# =============================================================================

class TtResNetBottleneck:
    """TTNN ResNet50 backbone with Bottleneck blocks.

    Standard 7x7 conv1 + maxpool, BN pre-folded into convolutions.
    Outputs multi-scale features [c2, c3, c4, c5] at strides [4, 8, 16, 32].
    """

    # ResNet50 layer config
    LAYERS = [3, 4, 6, 3]
    PLANES = [64, 128, 256, 512]
    STRIDES = [1, 2, 2, 2]

    def __init__(
        self,
        params: dict,
        conv_shapes: dict,
        device,
        batch_size: int = 6,
        input_height: int = 256,
        input_width: int = 704,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        style: str = "pytorch",
    ):
        self.device = device
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.out_indices = out_indices

        # Conv1: 7x7, stride 2, padding 3
        conv1_shapes = conv_shapes["conv1"]
        self.conv1 = Conv2dOp(
            params["conv1"], device,
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            batch_size=conv1_shapes["batch_size"],
            input_height=conv1_shapes["input_height"],
            input_width=conv1_shapes["input_width"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            activation_dtype=ttnn.bfloat16,
            act_block_h_override=64,
            deallocate_activation=True,
        )

        # Maxpool output dimensions
        self.maxpool_output_height = (input_height // 2 - 3 + 2 * 1) // 2 + 1  # (128-3+2)/2+1=64
        self.maxpool_output_width = (input_width // 2 - 3 + 2 * 1) // 2 + 1  # (352-3+2)/2+1=176

        # Build residual layers
        # Layer3 (1024ch) and Layer4 (2048ch) need BLOCK_SHARDED for conv3
        # and downsample to avoid L1 overflow. HEIGHT_SHARDED packs all channel
        # weights per core row which exceeds L1 for large channel counts.
        self.res_layers = []
        inplanes = 64
        for i in range(4):
            planes = self.PLANES[i]
            stride = self.STRIDES[i]
            num_blocks = self.LAYERS[i]

            # Collect shapes for this layer
            layer_shapes = self._collect_layer_shapes(conv_shapes, i + 1, num_blocks)

            # Use block sharding for deeper layers with large output channels
            use_block_shard = (i >= 2)  # layer3 (1024ch) and layer4 (2048ch)

            layer = ResLayer(
                params=params[f"layer{i + 1}"],
                shapes=layer_shapes,
                device=device,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                style=style,
                activation_dtype=ttnn.bfloat8_b if i == 1 else ttnn.bfloat16,
                conv3_block_sharded=use_block_shard,
                downsample_block_sharded=use_block_shard,
            )
            self.res_layers.append(layer)
            inplanes = planes * Bottleneck.expansion

    def _collect_layer_shapes(self, conv_shapes: dict, layer_idx: int, num_blocks: int) -> dict:
        """Collect conv shapes for all blocks in a layer."""
        shapes = {}
        prefix = f"layer{layer_idx}"

        # Block 0 shapes
        for conv_name in ["conv1", "conv2", "conv3"]:
            key = f"{prefix}.0.{conv_name}"
            if key in conv_shapes:
                shapes[conv_name] = conv_shapes[key]

        ds_key = f"{prefix}.0.downsample"
        if ds_key in conv_shapes:
            shapes["downsample"] = conv_shapes[ds_key]

        # Blocks 1+ shapes
        for j in range(1, num_blocks):
            for conv_name in ["conv1", "conv2", "conv3"]:
                key = f"{prefix}.{j}.{conv_name}"
                if key in conv_shapes:
                    shapes[f"block{j}.{conv_name}"] = conv_shapes[key]

        return shapes

    def __call__(self, x: ttnn.Tensor) -> List[ttnn.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor in TTNN format (1, 1, N*H*W, C) with NHWC layout flattened.

        Returns:
            List of feature map tensors at selected out_indices.
        """
        logger.debug("==== conv1 (7x7, stride 2)")
        x, out_h, out_w = self.conv1(x)

        # Convert to interleaved for maxpool
        x = ttnn.sharded_to_interleaved(x)
        x = ttnn.add(x, 0.0, dtype=ttnn.bfloat8_b)

        logger.debug(f"==== maxpool (3x3, stride 2), input: {out_h}x{out_w}")
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=out_h,
            input_w=out_w,
            channels=64,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ceil_mode=False,
        )

        # Run residual layers and collect outputs
        outs = []
        for i, layer in enumerate(self.res_layers):
            logger.debug(f"==== layer{i + 1}")
            x = layer(x)
            if i == 0:
                # Convert to bfloat8_b after layer1 for memory efficiency
                x = ttnn.add(x, 0.0, dtype=ttnn.bfloat8_b)
            if i in self.out_indices:
                outs.append(x)

        return outs


# =============================================================================
# Factory Function
# =============================================================================

def create_tt_resnet_bottleneck(
    torch_model: nn.Module,
    device,
    batch_size: int = 6,
    input_height: int = 256,
    input_width: int = 704,
    out_indices: Tuple[int, ...] = (0, 1, 2, 3),
    style: str = "pytorch",
) -> Tuple[TtResNetBottleneck, dict]:
    """Create a TTNN ResNet50 Bottleneck model from a PyTorch ResNet50.

    Folds BN into conv weights, infers conv shapes, and constructs the TTNN model.
    """
    torch_model.eval()

    # Step 1: Infer conv shapes via forward pass
    sample_input = torch.randn(batch_size, 3, input_height, input_width)
    conv_shapes = infer_conv_shapes(torch_model, sample_input)

    # Step 2: Preprocess weights (fold BN)
    params = preprocess_resnet50_parameters(torch_model)

    # Step 3: Build TTNN model
    tt_model = TtResNetBottleneck(
        params=params,
        conv_shapes=conv_shapes,
        device=device,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        out_indices=out_indices,
        style=style,
    )

    return tt_model, params
