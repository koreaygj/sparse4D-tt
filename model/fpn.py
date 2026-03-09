# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# FPN (Feature Pyramid Network) for TT Devices
#
# Ported from mmdet.models.necks.fpn.FPN for Sparse4D pipeline.
# Takes ResNet50 backbone outputs [c2, c3, c4, c5] and produces
# multi-scale feature maps [p2, p3, p4, p5] with unified 256 channels.
#
# Sparse4D config:
#   in_channels  = [256, 512, 1024, 2048]
#   out_channels = 256
#   num_outs     = 4
#   start_level  = 0
#   add_extra_convs = "on_output" (not used since num_outs == num_levels)
#   relu_before_extra_convs = True
#
# Forward flow:
#   1. Lateral 1x1 convs: reduce channels to 256
#   2. Top-down path: upsample + add (nearest 2x)
#   3. FPN 3x3 convs: refine each level
#
# TT-NN layout notes:
#   - ResNet50 outputs are in flattened format: [1, 1, N*H*W, C]
#   - conv2d expects [1, 1, N*H*W, C] (height-sharded or interleaved)
#   - upsample expects NHWC: [N, H, W, C]
#   - Need reshape between conv2d flattened format and upsample NHWC format
# =============================================================================

from typing import Dict, List, Tuple

import torch
import ttnn
from loguru import logger


class FPN:
    def __init__(
        self,
        device,
        parameters,
        batch_size: int,
        in_channels: List[int],
        out_channels: int,
        model_config: Dict,
        input_spatial_shapes: List[Tuple[int, int]],
    ) -> None:
        """
        Args:
            device: TT device
            parameters: Preprocessed model parameters (weights/biases)
            batch_size: Number of images (e.g. 6 cameras)
            in_channels: Channel dims from backbone [256, 512, 1024, 2048]
            out_channels: Unified output channels (256)
            model_config: Dict with WEIGHTS_DTYPE, ACTIVATIONS_DTYPE, MATH_FIDELITY
            input_spatial_shapes: [(H,W) for each level] e.g. [(64,176),(32,88),(16,44),(8,22)]
        """
        self.device = device
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_config = model_config
        self.input_spatial_shapes = input_spatial_shapes
        self.num_levels = len(in_channels)

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=model_config["MATH_FIDELITY"],
            # for fast math (should change if need to more accruarcy)
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config = compute_kernel_config

        # Store weight/bias tensors for lateral convs (1x1)
        self.lateral_weights = []
        self.lateral_biases = []
        for i in range(self.num_levels):
            self.lateral_weights.append(parameters.lateral_convs[i].conv.weight)
            self.lateral_biases.append(parameters.lateral_convs[i].conv.bias)

        # Store weight/bias tensors for fpn convs (3x3)
        self.fpn_weights = []
        self.fpn_biases = []
        for i in range(self.num_levels):
            self.fpn_weights.append(parameters.fpn_convs[i].conv.weight)
            self.fpn_biases.append(parameters.fpn_convs[i].conv.bias)

    def run(self, features: List[ttnn.Tensor], device) -> List[ttnn.Tensor]:
        """
        Args:
            features: [c2, c3, c4, c5] from ResNet50 backbone
                      Each in flattened format [1, 1, N*H*W, C] on DRAM
            device: TT device

        Returns:
            [p2, p3, p4, p5] in flattened format [1, 1, N*H*W, 256]
        """
        assert len(features) == self.num_levels

        # ==== Step 1: Lateral 1x1 convs (channel reduction to 256) ====
        laterals = []
        lateral_spatial = []
        for i in range(self.num_levels):
            h, w = self.input_spatial_shapes[i]
            logger.debug(
                f"==== FPN lateral conv {i}: in_ch={self.in_channels[i]}, spatial={h}x{w}"
            )

            x = features[i]
            # Ensure on DRAM interleaved for conv2d input
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

            # Use HEIGHT_SHARDED for large spatial, skip for small to avoid L1 clash
            total_rows = self.batch_size * h * w
            use_sharding = total_rows >= 128
            conv_config = ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_sharding else None
                ),
                reshard_if_not_optimal=True,
            )

            [x, [out_h, out_w], [self.lateral_weights[i], self.lateral_biases[i]]] = (
                ttnn.conv2d(
                    input_tensor=x,
                    weight_tensor=self.lateral_weights[i],
                    bias_tensor=self.lateral_biases[i],
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels,
                    device=device,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    batch_size=self.batch_size,
                    input_height=h,
                    input_width=w,
                    conv_config=conv_config,
                    compute_config=self.compute_kernel_config,
                    dtype=self.model_config["ACTIVATIONS_DTYPE"],
                    return_output_dim=True,
                    return_weights_and_bias=True,
                )
            )

            laterals.append(x)
            lateral_spatial.append((out_h, out_w))

        # ==== Step 2: Top-down pathway (upsample nearest 2x + add) ====
        # Process from top (smallest) to bottom (largest)
        for i in range(self.num_levels - 1, 0, -1):
            h_lower, w_lower = lateral_spatial[i - 1]
            h_upper, w_upper = lateral_spatial[i]
            logger.debug(
                f"==== FPN top-down: level {i}({h_upper}x{w_upper}) -> level {i - 1}({h_lower}x{w_lower})"
            )

            upper = laterals[i]

            # Move to DRAM interleaved + ROW_MAJOR for reshape/upsample
            upper = ttnn.to_memory_config(upper, ttnn.DRAM_MEMORY_CONFIG)
            upper = ttnn.to_layout(upper, ttnn.ROW_MAJOR_LAYOUT)

            # Reshape from flattened [1, 1, N*H*W, C] to NHWC [N, H, W, C] for upsample
            upper = ttnn.reshape(
                upper, (self.batch_size, h_upper, w_upper, self.out_channels)
            )

            # Upsample nearest 2x (NHWC format)
            scale_h = h_lower // h_upper
            scale_w = w_lower // w_upper
            upper = ttnn.upsample(upper, (scale_h, scale_w))

            # Reshape back to flattened [1, 1, N*H*W, C]
            upper = ttnn.reshape(
                upper, (1, 1, self.batch_size * h_lower * w_lower, self.out_channels)
            )

            # Back to DRAM interleaved + TILE for add
            upper = ttnn.sharded_to_interleaved(
                upper, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            upper = ttnn.to_layout(upper, ttnn.TILE_LAYOUT)

            # Element-wise add
            lower = laterals[i - 1]
            lower = ttnn.to_memory_config(lower, ttnn.DRAM_MEMORY_CONFIG)
            laterals[i - 1] = ttnn.add(lower, upper)

            ttnn.deallocate(lower)
            ttnn.deallocate(upper)

        # ==== Step 3: FPN 3x3 convs (output refinement) ====
        outs = []
        for i in range(self.num_levels):
            h, w = lateral_spatial[i]
            logger.debug(f"==== FPN output conv {i}: 256ch, spatial={h}x{w}")

            x = laterals[i]
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

            total_rows = self.batch_size * h * w
            use_sharding = total_rows >= 128
            conv_config = ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if use_sharding else None
                ),
                reshard_if_not_optimal=True,
            )

            [x, [out_h, out_w], [self.fpn_weights[i], self.fpn_biases[i]]] = (
                ttnn.conv2d(
                    input_tensor=x,
                    weight_tensor=self.fpn_weights[i],
                    bias_tensor=self.fpn_biases[i],
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    device=device,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    batch_size=self.batch_size,
                    input_height=h,
                    input_width=w,
                    conv_config=conv_config,
                    compute_config=self.compute_kernel_config,
                    dtype=self.model_config["ACTIVATIONS_DTYPE"],
                    return_output_dim=True,
                    return_weights_and_bias=True,
                )
            )

            # Store on DRAM for downstream consumption
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            outs.append(x)

            ttnn.deallocate(laterals[i])

        return outs


def preprocess_fpn_parameters(model, *, dtype=ttnn.bfloat16):
    """
    Preprocess PyTorch FPN parameters for TT-NN.

    Args:
        model: PyTorch FPN model (mmdet FPN instance)
        dtype: Target dtype for weights

    Returns:
        Dict-like structure with lateral_convs[i].conv.{weight, bias}
                                 and fpn_convs[i].conv.{weight, bias}
    """
    parameters = {}

    # Lateral convs (1x1, no norm, no activation)
    lateral_params = []
    for i, lc in enumerate(model.lateral_convs):
        conv = lc.conv
        weight = conv.weight  # [out_ch, in_ch, 1, 1]
        bias = conv.bias  # [out_ch]
        lateral_params.append(
            {
                "weight": ttnn.from_torch(weight),
                "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1)),
            }
        )

    # FPN convs (3x3, no norm, no activation)
    fpn_params = []
    for i, fc in enumerate(model.fpn_convs):
        conv = fc.conv
        weight = conv.weight  # [out_ch, in_ch, 3, 3]
        bias = conv.bias  # [out_ch]
        fpn_params.append(
            {
                "weight": ttnn.from_torch(weight),
                "bias": ttnn.from_torch(bias.reshape(1, 1, 1, -1)),
            }
        )

    return _FPNParameters(lateral_params, fpn_params)


class _FPNParameters:
    """Simple container to mimic ttnn parameter access pattern."""

    def __init__(self, lateral_params, fpn_params):
        self.lateral_convs = [_ConvParams(p) for p in lateral_params]
        self.fpn_convs = [_ConvParams(p) for p in fpn_params]


class _ConvParams:
    def __init__(self, params):
        self.conv = _WeightBias(params["weight"], params["bias"])


class _WeightBias:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
