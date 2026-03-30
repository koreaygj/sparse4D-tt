# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# AsymmetricFFN for TT Devices
#
# Feed-forward network with asymmetric input/output dimensions.
# Used after DeformableFeatureAggregation (residual_mode="cat") in Sparse4D
# decoder to reduce 512-dim back to 256-dim.
#
# Forward flow:
#   1. pre_norm: LayerNorm(512)
#   2. Linear(512 → 1024) + ReLU
#   3. Linear(1024 → 256)
#   4. identity_fc: Linear(512 → 256) on original input
#   5. output = identity + projected
# =============================================================================

import torch
import ttnn


class AsymmetricFFN:
    """TT-NN implementation of AsymmetricFFN.

    Mirrors mmcv AsymmetricFFN from Sparse4D blocks.py.
    Dropout is skipped (inference only).
    """

    def __init__(
        self,
        device,
        parameters: dict,
        in_channels: int = 512,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
    ) -> None:
        self.device = device
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self._hifi_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False,
        )

        # pre_norm: LayerNorm(in_channels)
        self.pre_norm_weight = self._to_device_1d(parameters["pre_norm_weight"])
        self.pre_norm_bias = self._to_device_1d(parameters["pre_norm_bias"])

        # layers[0][0]: Linear(in_channels → feedforward_channels)
        self.fc1_weight = self._to_device(parameters["fc1_weight"])
        self.fc1_bias = self._to_device_bias(parameters["fc1_bias"])

        # layers[1]: Linear(feedforward_channels → embed_dims)
        self.fc2_weight = self._to_device(parameters["fc2_weight"])
        self.fc2_bias = self._to_device_bias(parameters["fc2_bias"])

        # identity_fc: Linear(in_channels → embed_dims) if in_channels != embed_dims
        if in_channels != embed_dims:
            self.identity_fc_weight = self._to_device(parameters["identity_fc_weight"])
            self.identity_fc_bias = self._to_device_bias(parameters["identity_fc_bias"])
            self.has_identity_fc = True
        else:
            self.has_identity_fc = False

    def _to_device(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return ttnn.from_torch(
            tensor.float(), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32
        )

    def _to_device_bias(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        return ttnn.from_torch(
            tensor.float(), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32
        )

    def _to_device_1d(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        return ttnn.from_torch(
            tensor.float(), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32
        )

    def run(
        self,
        x: ttnn.Tensor,
        bs: int,
        num_tokens: int,
    ) -> ttnn.Tensor:
        """Forward pass of AsymmetricFFN.

        Args:
            x: [bs, num_tokens, in_channels] on device (TILE)
            bs: batch size
            num_tokens: number of tokens (e.g. 900 anchors)

        Returns:
            output: [bs, num_tokens, embed_dims] on device (TILE)
        """
        # Flatten for linear ops: [1, 1, bs*num_tokens, in_channels]
        x_flat = ttnn.reshape(x, (1, 1, bs * num_tokens, self.in_channels))

        # --- 1. Pre-norm (LayerNorm) ---
        normed = ttnn.layer_norm(
            x_flat, weight=self.pre_norm_weight, bias=self.pre_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=self._hifi_compute_config,
        )

        # --- 2. Identity path (use normed input, matching mmcv AsymmetricFFN) ---
        # mmcv default: identity=None → identity = pre_norm(x), then identity_fc(identity)
        if self.has_identity_fc:
            identity = ttnn.linear(
                normed, self.identity_fc_weight, bias=self.identity_fc_bias,
                compute_kernel_config=self._hifi_compute_config,
            )
        else:
            identity = normed

        # --- 3. Linear(in_channels → feedforward_channels) + ReLU ---
        fc1_out = ttnn.linear(normed, self.fc1_weight, bias=self.fc1_bias, compute_kernel_config=self._hifi_compute_config)
        # Only deallocate normed if identity doesn't alias it (asymmetric case)
        if self.has_identity_fc:
            ttnn.deallocate(normed)

        relu_out = ttnn.relu(fc1_out)

        # --- 4. Linear(feedforward_channels → embed_dims) ---
        projected = ttnn.linear(relu_out, self.fc2_weight, bias=self.fc2_bias, compute_kernel_config=self._hifi_compute_config)

        # Sync + deallocate large intermediates (1024-dim tensors)
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(fc1_out)
        ttnn.deallocate(relu_out)

        # --- 5. Residual ---
        output = ttnn.add(identity, projected)

        # Sync + deallocate
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(projected)
        ttnn.deallocate(identity)

        output = ttnn.reshape(output, (bs, num_tokens, self.embed_dims))

        return output

def preprocess_ffn_parameters(pt_ffn_layer) -> dict:
    """Extract parameters from mmcv AsymmetricFFN.

    AsymmetricFFN structure (num_fcs=2, in_channels=512, embed_dims=256):
        pre_norm: LayerNorm(512)
        layers: Sequential(
            Sequential(Linear(512, 1024), ReLU, Dropout),  # layers[0]
            Linear(1024, 256),                              # layers[1]
            Dropout,                                        # layers[2]
        )
        identity_fc: Linear(512, 256)  (when in_channels != embed_dims)

    Args:
        pt_ffn_layer: mmcv AsymmetricFFN instance

    Returns:
        dict of torch tensors (weights already transposed for ttnn.linear)
    """
    params = {}

    # pre_norm
    if hasattr(pt_ffn_layer, 'pre_norm') and pt_ffn_layer.pre_norm is not None:
        params["pre_norm_weight"] = pt_ffn_layer.pre_norm.weight.data.clone()
        params["pre_norm_bias"] = pt_ffn_layer.pre_norm.bias.data.clone()

    # layers[0] is Sequential(Linear, ReLU, Dropout) → Linear is [0][0]
    fc1 = pt_ffn_layer.layers[0][0]
    params["fc1_weight"] = fc1.weight.data.clone().t()
    params["fc1_bias"] = fc1.bias.data.clone()

    # layers[1] is Linear(feedforward_channels, embed_dims)
    fc2 = pt_ffn_layer.layers[1]
    params["fc2_weight"] = fc2.weight.data.clone().t()
    params["fc2_bias"] = fc2.bias.data.clone()

    # identity_fc (if in_channels != embed_dims)
    if hasattr(pt_ffn_layer, 'identity_fc') and not isinstance(
        pt_ffn_layer.identity_fc, torch.nn.Identity
    ):
        params["identity_fc_weight"] = pt_ffn_layer.identity_fc.weight.data.clone().t()
        params["identity_fc_bias"] = pt_ffn_layer.identity_fc.bias.data.clone()

    return params
