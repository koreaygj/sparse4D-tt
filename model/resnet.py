# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# ResNet50 Backbone for TT Devices (forked from tt-metal ttnn_functional_resnet50)
#
# This module implements ResNet50 using TTNN ops, producing FPN feature maps
# (c2, c3, c4, c5) at strides 4, 8, 16, 32 for downstream detection heads.
#
# Device Grid Reference:
#   N150 (WH_B0 unharvested): 8x8 = 64 cores
#   N300 (WH_B0 harvested):   8x7 = 56 cores
#   P150 (BH unharvested):   10x13 = 130 cores
#   P100 (BH harvested):     varies < 130 cores
#
# Note: is_wormhole_b0() returns True for BOTH N150 and N300.
#       It only checks arch name, NOT the actual grid size.
#       Use device.compute_with_storage_grid_size() to get actual grid.
# =============================================================================

import math
from typing import List

import torch
import ttnn
from loguru import logger
from models.common.utility_functions import (
    _nearest_y,
    is_blackhole,
    is_n300,
    is_wormhole_b0,
    nearest_32,
)
from models.demos.vision.classification.resnet50.ttnn_resnet.tt.ttnn_functional_resnet50_model_utils import (
    is_blackhole_p100,
)


# =============================================================================
# ResnetLinear: Creates a closure for the final FC layer (1000-class output).
# Uses 1D multicast matmul on an 8x4 = 32 core grid.
# This grid (32 cores) fits all device variants safely.
# =============================================================================
def ResnetLinear(
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    output_mem_config,
    model_config,
    compute_kernel_config,
):
    """
    Returns a function for linear operation in resnet with bias.
    """

    matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),  # 32 cores — safe for all devices
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    weight = weight.reshape(weight.shape.to_rank(4))
    bias = bias.reshape(bias.shape.to_rank(4))

    def linear_(act):
        output = ttnn.linear(
            act,
            weight,
            bias=bias,
            program_config=matmul_config,
            memory_config=output_mem_config,
            dtype=model_config["ACTIVATIONS_DTYPE"],
            compute_kernel_config=compute_kernel_config,
        )
        return output

    return linear_


# =============================================================================
# resnet50Bottleneck: A single bottleneck block (conv1x1 → conv3x3 → conv1x1 + residual)
#
# Each block does:
#   1. conv1: 1x1 (channel reduction)
#   2. conv2: 3x3 (spatial, with stride for downsampling blocks)
#   3. conv3: 1x1 (channel expansion to 4x planes)
#   4. Downsample shortcut (optional, only first block of each layer)
#   5. Residual add + ReLU
#
# Sharding strategy is controlled by caller via height_sharding param:
#   - HEIGHT_SHARDED: rows distributed across cores (used for layers 1-2)
#   - BLOCK_SHARDED:  2D tile distribution (used for layers 3-4)
#
# The conv2d op handles its own internal sharding via reshard_if_not_optimal.
# The bottleneck itself has NO hardcoded core grids — the problem grids are
# all in the resnet50 class's run() method between layer transitions.
# =============================================================================
class resnet50Bottleneck:
    expansion: int = 4

    def __init__(self, parameters, downsample, stride, model_config) -> None:
        # init is just to pre-process pytorch weights and bias tensors
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 1

        self.conv2_weight_tensor = parameters.conv2.weight
        self.conv2_bias_tensor = parameters.conv2.bias
        self.conv2_input_channels = self.conv2_weight_tensor.shape[1]
        self.conv2_output_channels = self.conv2_weight_tensor.shape[0]
        self.conv2_stride = 2 if downsample else 1
        assert self.conv2_weight_tensor.shape[2] == 3

        self.conv3_weight_tensor = parameters.conv3.weight
        self.conv3_bias_tensor = parameters.conv3.bias
        self.conv3_input_channels = self.conv3_weight_tensor.shape[1]
        self.conv3_output_channels = self.conv3_weight_tensor.shape[0]
        assert self.conv3_weight_tensor.shape[2] == 1

        self.downsample = downsample
        self.stride = stride
        if downsample:
            self.ds_conv_weight_tensor = parameters.downsample.weight
            self.ds_conv_bias_tensor = parameters.downsample.bias
            self.ds_conv_input_channels = self.ds_conv_weight_tensor.shape[1]
            self.ds_conv_output_channels = self.ds_conv_weight_tensor.shape[0]
            assert self.ds_conv_weight_tensor.shape[2] == 1
        self.model_config = model_config
        return

    def run_downsample_if_req(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        packer_l1_accum_enabled=True,
    ):
        if self.downsample:
            logger.debug(f"Running downsample")
            conv_kwargs = {
                "in_channels": self.ds_conv_input_channels,
                "out_channels": self.ds_conv_output_channels,
                "batch_size": batch_size,
                "input_height": input_height,
                "input_width": input_width,
                "kernel_size": (1, 1),
                "stride": (self.stride, self.stride),
                "padding": (0, 0),
                "dilation": (1, 1),
                "groups": 1,
                "device": device,
                "conv_config": ttnn.Conv2dConfig(
                    weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                    shard_layout=(
                        ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                        if height_sharding and input_height != 28
                        else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                    ),
                    deallocate_activation=True,
                    reallocate_halo_output=False,
                    reshard_if_not_optimal=reshard_if_not_optimal,
                    enable_act_double_buffer=True
                    if not (is_blackhole_p100(device) and batch_size > 16)
                    else False,
                    enable_weights_double_buffer=True if input_width < 56 else False,
                    full_inner_dim=True,
                    enable_activation_reuse=True
                    if height_sharding and self.stride == 1
                    else False,
                ),
            }

            ds_out, [self.ds_conv_weight_tensor, self.ds_conv_bias_tensor] = (
                ttnn.conv2d(
                    input_tensor=x,
                    weight_tensor=self.ds_conv_weight_tensor,
                    bias_tensor=self.ds_conv_bias_tensor,
                    **conv_kwargs,
                    compute_config=ttnn.init_device_compute_kernel_config(
                        device.arch(),
                        math_fidelity=self.model_config["MATH_FIDELITY"],
                        packer_l1_acc=packer_l1_accum_enabled,
                    ),
                    return_output_dim=False,
                    return_weights_and_bias=True,
                    dtype=self.model_config["ACTIVATIONS_DTYPE"],
                )
            )
        else:
            ds_out = x
        return ds_out

    def __call__(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,  # if True, conv2d will reshard input if layout is suboptimal
        height_sharding=None,  # True=HEIGHT_SHARDED, False/None=BLOCK_SHARDED
        packer_l1_acc=True,
        layer_module=None,  # string like "layer2_module3" for per-module overrides
    ):
        logger.debug(
            f"==== Running {batch_size}, {input_height}, {input_width}, {self.conv1_input_channels}, {self.conv1_output_channels}"
        )

        ds_input_height = input_height
        ds_input_width = input_width

        # conv1 is 1x1 conv
        logger.debug(f"Running conv1")
        conv_kwargs_1 = {
            "in_channels": self.conv1_input_channels,
            "out_channels": self.conv1_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                    if height_sharding
                    else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
            ),
        }

        (
            out,
            [input_height, input_width],
            [self.conv1_weight_tensor, self.conv1_bias_tensor],
        ) = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            bias_tensor=self.conv1_bias_tensor,
            **conv_kwargs_1,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        act_block_h_override = 0
        ds_out = None

        logger.debug(f"Running conv2")

        conv_kwargs_2 = {
            "in_channels": self.conv2_input_channels,
            "out_channels": self.conv2_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (3, 3),
            "stride": (self.stride, self.stride),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                deallocate_activation=True,
                reallocate_halo_output=False,
                act_block_h_override=act_block_h_override,
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                    if height_sharding
                    else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
                full_inner_dim=True,
                enable_activation_reuse=True
                if height_sharding and self.stride == 1
                else False,
            ),
        }

        if is_blackhole():
            if layer_module == "layer1_module3":
                conv_kwargs_2["conv_config"].act_block_h_override = 16 * 32
            if batch_size == 32 and is_blackhole_p100(device):
                if (
                    layer_module == "layer1_module2"
                    or layer_module == "layer1_module3"
                    or layer_module == "layer2_module1"
                ):
                    conv_kwargs_2["conv_config"].act_block_h_override = 32

        if is_wormhole_b0():
            if layer_module == "layer1_module2" or layer_module == "layer1_module3":
                conv_kwargs_2["conv_config"].act_block_h_override = 14 * 32

        (
            out,
            [input_height, input_width],
            [self.conv2_weight_tensor, self.conv2_bias_tensor],
        ) = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=self.conv2_weight_tensor,
            bias_tensor=self.conv2_bias_tensor,
            **conv_kwargs_2,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        # conv3 is 1x1 conv
        logger.debug(f"Running conv3")
        conv_kwargs_3 = {
            "in_channels": self.conv3_input_channels,
            "out_channels": self.conv3_output_channels,
            "batch_size": batch_size,
            "input_height": input_height,
            "input_width": input_width,
            "kernel_size": (1, 1),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                    if height_sharding
                    else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=reshard_if_not_optimal,
                deallocate_activation=True,
            ),
        }

        out, [self.conv3_weight_tensor, self.conv3_bias_tensor] = ttnn.conv2d(
            input_tensor=out,
            weight_tensor=self.conv3_weight_tensor,
            bias_tensor=self.conv3_bias_tensor,
            **conv_kwargs_3,
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.model_config["MATH_FIDELITY"],
                packer_l1_acc=packer_l1_acc,
            ),
            return_output_dim=False,
            return_weights_and_bias=True,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        ds_out = self.run_downsample_if_req(
            x,
            device,
            batch_size,
            ds_input_height,
            ds_input_width,
            reshard_if_not_optimal,
            height_sharding,
            packer_l1_accum_enabled=packer_l1_acc,
        )

        if ds_out.memory_config() != out.memory_config():
            ds_out = ttnn.to_memory_config(ds_out, out.memory_config())

        # underscore version is in_place = True
        out = ttnn.add_(
            out,
            ds_out,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
        )
        ttnn.deallocate(ds_out)
        return out, input_height, input_width


# =============================================================================
# resnet50: Full ResNet50 model producing FPN feature maps [c2, c3, c4, c5]
#
# Architecture: Input → Fold → Conv1 → MaxPool → Layer1-4 → [c2, c3, c4, c5]
#
# __init__ handles:
#   - Weight loading and layer construction
#   - Conv1 configuration (arch-specific act_block_h overrides)
#   - Fold operation grid setup (batch-size dependent)
#   - Fold memory config (uses device.compute_with_storage_grid_size())
#
# run() handles:
#   - Forward pass through all layers
#   - Manual resharding between layer transitions (⚠️ hardcoded grids here)
#   - Returns [c2, c3, c4, c5] feature maps at strides [4, 8, 16, 32]
# =============================================================================
class resnet50:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
        input_shape,  # (C, H, W) of raw input image
        kernel_size,  # fold kernel size (padding amount)
        stride,  # fold stride (downsampling factor)
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    ) -> None:
        super().__init__()
        layers = [3, 4, 6, 3]  # ResNet50: [3, 4, 6, 3] bottleneck blocks
        conv_input_face_shape_hw = [224, 224]
        self.device = device
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        self.batch_size = batch_size
        self.model_config = model_config
        self.inplanes = 64
        self.final_output_mem_config = final_output_mem_config
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=model_config["MATH_FIDELITY"],
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 4

        # ---- Build 4 residual layers (each is a list of Bottleneck blocks) ----
        # layer1: 3 blocks, 64→256 ch,  stride=1, output=56×56  (stride 4 from input)
        # layer2: 4 blocks, 256→512 ch, stride=2, output=28×28  (stride 8)
        # layer3: 6 blocks, 512→1024 ch, stride=2, output=14×14 (stride 16)
        # layer4: 3 blocks, 1024→2048 ch, stride=2, output=7×7  (stride 32)
        self.layer1 = self._make_layer(
            parameters=parameters.layer1,
            planes=64,
            blocks=layers[0],
            stride=1,
            model_config=model_config,
        )
        self.layer2 = self._make_layer(
            parameters=parameters.layer2,
            planes=128,
            blocks=layers[1],
            stride=2,
            model_config=model_config,
        )
        self.layer3 = self._make_layer(
            parameters=parameters.layer3,
            planes=256,
            blocks=layers[2],
            stride=2,
            model_config=model_config,
        )
        self.layer4 = self._make_layer(
            parameters=parameters.layer4,
            planes=512,
            blocks=layers[3],
            stride=2,
            model_config=model_config,
        )

        # ---- Unroll all modules into named attributes for explicit control ----
        # This allows per-module overrides (e.g., act_block_h) in run().
        # Only [3, 4, 6, 3] is supported — no dynamic layer counts.
        assert layers == [3, 4, 6, 3]
        self.layer1_module1 = self.layer1[0]
        self.layer1_module2 = self.layer1[1]
        self.layer1_module3 = self.layer1[2]

        self.layer2_module1 = self.layer2[0]
        self.layer2_module2 = self.layer2[1]
        self.layer2_module3 = self.layer2[2]
        self.layer2_module4 = self.layer2[3]

        self.layer3_module1 = self.layer3[0]
        self.layer3_module2 = self.layer3[1]
        self.layer3_module3 = self.layer3[2]
        self.layer3_module4 = self.layer3[3]
        self.layer3_module5 = self.layer3[4]
        self.layer3_module6 = self.layer3[5]

        self.layer4_module1 = self.layer4[0]
        self.layer4_module2 = self.layer4[1]
        self.layer4_module3 = self.layer4[2]

        self.fc = ResnetLinear(
            weight=ttnn.to_device(parameters.fc.weight, device),
            bias=ttnn.to_device(parameters.fc.bias, device),
            output_mem_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            model_config=model_config,
            compute_kernel_config=compute_kernel_config,
        )  # num_classes = 1000

        # ---- Conv1 configuration ----
        # act_block_h_override controls the activation block height for conv1.
        # This affects how activations are tiled across cores.
        # Larger values = more work per core = fewer cores needed.
        act_block_h_override = 0

        if is_wormhole_b0():
            act_block_h_override = 1568  # 49 * 32 = 1568

        if is_blackhole() and self.batch_size == 32:
            act_block_h_override = 32 * 32 if is_blackhole_p100(device) else 49 * 32

        self.conv1_config = ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=dealloc_input,
            act_block_h_override=act_block_h_override,
            enable_act_double_buffer=True,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
            # otherwise act block h is not big enough for the reuse
            enable_activation_reuse=(
                not is_wormhole_b0() or device.get_num_devices() <= 8
            ),
        )
        self.conv1_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            packer_l1_acc=True,
        )
        if is_wormhole_b0():
            # Issue #13145: Temp workaround for Galaxy to avoid hangs
            if device.get_num_devices() > 8:
                self.conv1_config.act_block_h_override = 64
            else:
                self.conv1_config.act_block_h_override = 49 * 32

        self.conv1_kernel_size = (4, 4)
        self.conv1_stride = (1, 1)
        self.conv1_padding = (0, 0)
        self.conv1_input_height = 115
        self.conv1_input_width = 115
        self.conv1_output_height = (
            (
                self.conv1_input_height
                - self.conv1_kernel_size[0]
                + 2 * self.conv1_padding[0]
            )
            // self.conv1_stride[0]
        ) + 1
        self.conv1_output_width = (
            (
                self.conv1_input_width
                - self.conv1_kernel_size[1]
                + 2 * self.conv1_padding[1]
            )
            // self.conv1_stride[1]
        ) + 1

        # ---- Fold parameters ----
        # Fold is a preprocessing step that rearranges spatial data into channels,
        # effectively doing a stride-based spatial-to-depth transform.
        # Input: (N, C, H, W) → Output: (N, H/stride, W/stride, C * stride^2)
        # This replaces the standard 7x7 stride-2 conv with a 4x4 stride-1 conv
        # on the folded input, which is more efficient on TT hardware.
        self.fold_stride_h = stride
        self.fold_stride_w = stride
        _, c, h, w = input_shape
        n = batch_size
        h += kernel_size * 2  # padding added to spatial dims
        w += kernel_size * 2
        C = _nearest_y(c, 4)  # pad channels to nearest multiple of 4
        self.fold_pad_c = C - c
        self.fold_pad_h = kernel_size
        self.fold_pad_w = kernel_size
        self.fold_output_shape = (
            n,
            h // self.fold_stride_h,
            w // self.fold_stride_w,
            C * (self.fold_stride_h * self.fold_stride_w),
        )
        # ---- Fold compute grid ----
        # Determines how many cores execute the fold operation.
        # Use actual device grid to avoid exceeding available cores.
        compute_grid = device.compute_with_storage_grid_size()
        num_cores_x = compute_grid.x
        num_cores_y = compute_grid.y
        if self.batch_size == 16:
            num_cores_x = compute_grid.x
            num_cores_y = compute_grid.y
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1),
                    )
                }
            )
        elif self.batch_size == 20:
            if is_wormhole_b0():
                num_cores_x = 8
                num_cores_y = (
                    5  # 40 cores — safe for all WH variants (N150=64, N300=56)
                )
            elif is_blackhole():
                num_cores_x = 10
                num_cores_y = 8  # 80 cores — safe for BH (P150=130, P100<130 but >=80)
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1),
                    )
                }
            )
        elif self.batch_size == 32:
            # BH only: 13×9 + partial row = 128 cores (fits P150's 130)
            core_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 8)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(10, 9)),
                }
            )
            if is_blackhole_p100(device):
                # P100 has fewer cores, fall back to 8×8=64
                core_grid = ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}
                )
            self.fold_compute_grid_size = core_grid
        else:
            # Default: use full device grid for any other batch size
            self.fold_compute_grid_size = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1),
                    )
                }
            )

        conv_dummy_tensor = torch.rand((self.fold_output_shape), dtype=torch.bfloat16)
        conv_dummy_tensor = ttnn.from_torch(
            conv_dummy_tensor, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        # ---- Fold memory config (override_fold_mem_config) ----
        # This section CORRECTLY uses the device API to get the actual grid size.
        # device.compute_with_storage_grid_size() returns the real hardware grid:
        #   N150: (8, 8), N300: (8, 7), P150: (10, 13), P100: varies
        compute_grid = device.compute_with_storage_grid_size()

        # Calculate core grid
        if is_blackhole():
            # Override num cores to avoid padding issues
            nhw_ntiles = math.ceil(
                self.batch_size
                * self.conv1_output_height
                * self.conv1_output_width
                / 32
            )
            # Find closest largest divisor
            num_cores_target = compute_grid.x * compute_grid.y
            while nhw_ntiles % num_cores_target != 0:
                num_cores_target -= 1
            core_grid = ttnn.num_cores_to_corerangeset(
                num_cores_target, compute_grid, row_wise=True
            )
        else:
            # Use full grid
            core_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1),
                    )
                }
            )

        # Calculate shard dimensions
        input_channels_padded = (
            nearest_32(self.conv1_input_channels)
            if self.conv1_input_channels % 8 != 0
            else self.conv1_input_channels
        )
        if input_channels_padded % 8 != 0:
            input_channels_padded = ((input_channels_padded + 7) // 8) * 8

        tensor_height = (
            self.conv1_input_width * self.conv1_input_height * self.batch_size
        )
        tensor_width = input_channels_padded

        # Calculate shard shape for HEIGHT sharding
        num_cores = core_grid.num_cores()
        shard_height = math.ceil(tensor_height / num_cores)
        shard_width = tensor_width

        self.override_fold_mem_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, shard_height, shard_width),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def __del__(self):
        # Nothing to do
        pass

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        model_config=None,
    ) -> List[resnet50Bottleneck]:
        layers = []
        layers.append(
            resnet50Bottleneck(
                parameters=parameters[0],
                downsample=stride != 1
                or self.inplanes != planes * resnet50Bottleneck.expansion,
                stride=stride,
                model_config=model_config,
            )
        )
        self.inplanes = planes * resnet50Bottleneck.expansion
        for block_num in range(1, blocks):
            layers.append(
                resnet50Bottleneck(
                    parameters=parameters[block_num],
                    downsample=False,
                    stride=1,
                    model_config=model_config,
                )
            )
        return layers

    def __call__(self, input_tensor, device, ops_parallel_config) -> ttnn.Tensor:
        return self.run(
            input_tensor,
            device,
        )

    # =========================================================================
    # run(): Forward pass producing FPN feature maps [c2, c3, c4, c5]
    #
    # Data flow:
    #   Input → Fold → Conv1(4x4) → MaxPool(3x3,s2) → 56×56
    #   → [Reshard] → Layer1(×3) → c2 (stride 4, 256ch, 56×56)
    #   → Layer2(×4) → [Reshard] → c3 (stride 8, 512ch, 28×28)
    #   → Layer3(×6) → [Reshard] → c4 (stride 16, 1024ch, 14×14)
    #   → Layer4(×3) → c5 (stride 32, 2048ch, 7×7)
    #
    # ⚠️ HARDCODED CORE GRIDS in this method (all under is_wormhole_b0()):
    #   1. Post-maxpool reshard:  CoreGrid(x=8, y=7) — works on N300 by luck
    #   2. Layer2→3 reshard:     CoreGrid(x=8, y=8) — CRASHES on N300!
    #   3. Layer3→4 reshard:     CoreGrid(x=8, y=7) — works on N300 by luck
    # =========================================================================
    def run(self, input_tensor, device) -> ttnn.Tensor:
        logger.debug(f"==== fold on device")

        # ---- Stage 0: Fold (spatial-to-depth transform) ----
        fold_output_tensor = ttnn.fold(
            input_tensor,
            self.fold_stride_h,
            self.fold_stride_w,
            use_transpose_as_fold=True,
            padding=[
                self.fold_pad_h,
                self.fold_pad_h,
                self.fold_pad_w,
                self.fold_pad_w,
                0,
                self.fold_pad_c,
            ],
            grid_size=self.fold_compute_grid_size,
            override_memory_config=self.override_fold_mem_config,
        )
        n, c, h, w = fold_output_tensor.shape
        fold_output_tensor = ttnn.reshape(fold_output_tensor, (1, 1, n * c * h, w))

        ttnn.deallocate(input_tensor)

        logger.debug(f"==== first conv")

        # ---- Stage 1: Conv1 (4×4 kernel, stride 1, on folded input) ----
        # Input: folded tensor (batch*115*115, channels)
        # Output: (batch*112*112, 64) — 112×112 spatial
        conv_kwargs = {
            "in_channels": self.conv1_input_channels,
            "out_channels": self.conv1_output_channels,
            "batch_size": self.batch_size,
            "input_height": self.conv1_input_height,
            "input_width": self.conv1_input_width,
            "kernel_size": self.conv1_kernel_size,
            "stride": self.conv1_stride,
            "padding": self.conv1_padding,
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": self.conv1_config,
        }

        x, [x_height, x_width], [self.conv1_weight_tensor, self.conv1_bias_tensor] = (
            ttnn.conv2d(
                input_tensor=fold_output_tensor,
                weight_tensor=self.conv1_weight_tensor,
                bias_tensor=self.conv1_bias_tensor,
                **conv_kwargs,
                compute_config=self.conv1_compute_config,
                return_output_dim=True,
                return_weights_and_bias=True,
                dtype=self.model_config["ACTIVATIONS_DTYPE"],
            )
        )

        # ---- Stage 2: MaxPool (3×3, stride 2) ----
        # Input: 112×112 → Output: 56×56, 64 channels
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=x_height,
            input_w=x_width,
            channels=self.conv1_output_channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )

        x_height = 56
        x_width = 56

        # ---- Reshard #1: Post-maxpool → HEIGHT_SHARDED for Layer1 ----
        if is_wormhole_b0():
            compute_grid = device.compute_with_storage_grid_size()
            core_range_set = ttnn.CoreGrid(x=compute_grid.x, y=compute_grid.y)
            mem_config = ttnn.create_sharded_memory_config_(
                x.shape,
                core_range_set,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
            )
            x = ttnn.to_memory_config(x, mem_config)
            x = ttnn.to_layout(
                x, ttnn.TILE_LAYOUT, dtype=self.model_config["ACTIVATIONS_DTYPE"]
            )

        # ---- Layer 1: 3 bottleneck blocks (64→256ch, stride 1, 56×56) ----
        # Uses HEIGHT_SHARDED. Reshard only needed on Blackhole.
        logger.debug(f"==== Running layer 1 module 1")

        reshard = is_blackhole()  # BH needs explicit reshard; WH conv2d handles it
        height_shard = True  # Layer 1-2 use height sharding

        x, x_height, x_width = self.layer1_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module="layer1_module1",
        )

        logger.debug(f"==== Running layer 1 module 2")
        x, x_height, x_width = self.layer1_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer1_module2",
        )

        logger.debug(f"==== Running layer 1 module 3")
        x, x_height, x_width = self.layer1_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer1_module3",
        )

        reshard = False
        height_shard = True

        # ---- c2: Layer1 output, stride 4 from original input, 256ch, 56×56 ----
        # Save to DRAM to preserve buffer and free L1 for subsequent layers
        c2 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # ---- Layer 2: 4 bottleneck blocks (128→512ch, stride 2, 28×28) ----
        logger.debug(f"==== Running layer 2 module 1")
        x, x_height, x_width = self.layer2_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module="layer2_module1",
        )

        logger.debug(f"==== Running layer 2 module 2")
        x, x_height, x_width = self.layer2_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer2_module2",
        )

        logger.debug(f"==== Running layer 2 module 3")
        x, x_height, x_width = self.layer2_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer2_module3",
        )

        logger.debug(f"==== Running layer 2 module 4")
        x, x_height, x_width = self.layer2_module4(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer2_module4",
        )

        # ---- Reshard #2: Layer2 output → sharding for Layer3 ----
        # N300 (8×7=56 cores): block sharding overflows L1, use height sharding + reshard
        # N150 (8×8=64 cores): block sharding fits, use original path
        compute_grid = device.compute_with_storage_grid_size()
        n300_wh = is_wormhole_b0() and compute_grid.y < 8
        reshard = is_blackhole() or n300_wh
        height_shard = is_blackhole() or n300_wh
        if is_wormhole_b0() and not n300_wh:
            x = ttnn.to_memory_config(
                x,
                ttnn.create_sharded_memory_config(
                    x.shape,
                    ttnn.CoreGrid(x=compute_grid.x, y=compute_grid.y),
                    ttnn.ShardStrategy.BLOCK,
                ),
            )

        # ---- c3: Layer2 output, stride 8 from original input, 512ch, 28×28 ----
        c3 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # ---- Layer 3: 6 bottleneck blocks (256→1024ch, stride 2, 14×14) ----
        # On BH: uses height sharding + reshard. On WH: block sharding, no reshard.
        logger.debug(f"==== Running layer 3 module 1")
        x, x_height, x_width = self.layer3_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module="layer3_module1",
        )

        logger.debug(f"==== Running layer 3 module 2")
        x, x_height, x_width = self.layer3_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer3_module2",
        )

        logger.debug(f"==== Running layer 3 module 3")
        x, x_height, x_width = self.layer3_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer3_module3",
        )

        logger.debug(f"==== Running layer 3 module 4")
        x, x_height, x_width = self.layer3_module4(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer3_module4",
        )

        logger.debug(f"==== Running layer 3 module 5")
        x, x_height, x_width = self.layer3_module5(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer3_module5",
        )

        logger.debug(f"==== Running layer 3 module 6")
        x, x_height, x_width = self.layer3_module6(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer3_module6",
        )

        # ---- Reshard #3: Layer3 output → BLOCK_SHARDED for Layer4 ----
        # N300 was height-sharded in layer3, now switch back to block for layer4
        reshard = False
        height_shard = False

        if is_wormhole_b0():
            compute_grid = device.compute_with_storage_grid_size()
            block_mem_config = ttnn.create_sharded_memory_config(
                x.shape,
                ttnn.CoreGrid(x=compute_grid.x, y=compute_grid.y),
                ttnn.ShardStrategy.BLOCK,
            )
            x = ttnn.to_memory_config(x, block_mem_config)
        if is_blackhole():
            # BH uses explicit shard dimensions with 8×10=80 cores
            grid_size = (8, 10)
            block_mem_config = ttnn.create_sharded_memory_config_(
                [nearest_32(x.shape[2] // grid_size[1]), x.shape[3] // grid_size[0]],
                ttnn.CoreGrid(x=grid_size[0], y=grid_size[1]),
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
                use_height_and_width_as_shard_shape=True,
            )
            x = ttnn.to_memory_config(x, block_mem_config)

        # ---- c4: Layer3 output, stride 16 from original input, 1024ch, 14×14 ----
        c4 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # ---- Layer 4: 3 bottleneck blocks (512→2048ch, stride 2, 7×7) ----
        # Both WH and BH: block sharding, no reshard needed.
        logger.debug(f"==== Running layer 4 module 1")
        x, x_height, x_width = self.layer4_module1(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            reshard_if_not_optimal=reshard,
            height_sharding=height_shard,
            layer_module="layer4_module1",
        )

        logger.debug(f"==== Running layer 4 module 2")
        x, x_height, x_width = self.layer4_module2(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer4_module2",
        )

        logger.debug(f"==== Running layer 4 module 3")
        x, x_height, x_width = self.layer4_module3(
            x,
            device,
            self.batch_size,
            x_height,
            x_width,
            layer_module="layer4_module3",
        )

        # ---- c5: Layer4 output, stride 32 from original input, 2048ch, 7×7 ----
        c5 = ttnn.to_memory_config(x, self.final_output_mem_config)

        # Return FPN feature maps at strides [4, 8, 16, 32]
        return [c2, c3, c4, c5]
