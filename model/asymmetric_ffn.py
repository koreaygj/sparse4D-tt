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
        mesh_device=None,
    ) -> None:
        self.device = mesh_device if mesh_device is not None else device
        self._mesh_device = mesh_device
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self._hifi_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False, packer_l1_acc=False, math_approx_mode=True,
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
            # Fused identity+fc1: combined weight [in_channels, embed_dims+feedforward]
            import torch as _torch
            w_fused = _torch.cat([parameters["identity_fc_weight"], parameters["fc1_weight"]], dim=-1)
            b_id = parameters["identity_fc_bias"]
            b_fc1 = parameters["fc1_bias"]
            if b_id.dim() == 1:
                b_id = b_id.reshape(1, 1, 1, -1)
            if b_fc1.dim() == 1:
                b_fc1 = b_fc1.reshape(1, 1, 1, -1)
            b_fused = _torch.cat([b_id, b_fc1], dim=-1)
            self.w_identity_fc1 = self._to_device(w_fused)
            self.b_identity_fc1 = self._to_device_bias(b_fused.squeeze())
        else:
            self.has_identity_fc = False

    def _to_device(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

    def _to_device_bias(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

    def _to_device_1d(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

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
        x_flat = ttnn.reshape(x, (1, 1, bs * num_tokens, self.in_channels))
        normed = ttnn.layer_norm(x_flat, weight=self.pre_norm_weight, bias=self.pre_norm_bias,
                                  epsilon=1e-5, compute_kernel_config=self._hifi_compute_config)

        if self.has_identity_fc:
            # Fused identity_fc + fc1: one matmul [512, 1280] instead of [512, 256] + [512, 1024]
            fused_out = ttnn.linear(normed, self.w_identity_fc1, bias=self.b_identity_fc1,
                                    compute_kernel_config=self._hifi_compute_config)
            ttnn.deallocate(normed)
            identity = ttnn.slice(fused_out, [0, 0, 0, 0],
                                  [1, 1, bs * num_tokens, self.embed_dims])
            fc1_out = ttnn.slice(fused_out, [0, 0, 0, self.embed_dims],
                                 [1, 1, bs * num_tokens, self.embed_dims + self.feedforward_channels])
            ttnn.deallocate(fused_out)
        else:
            identity = normed
            fc1_out = ttnn.linear(normed, self.fc1_weight, bias=self.fc1_bias,
                                  compute_kernel_config=self._hifi_compute_config)

        relu_out = ttnn.relu(fc1_out)
        projected = ttnn.linear(relu_out, self.fc2_weight, bias=self.fc2_bias, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(fc1_out); ttnn.deallocate(relu_out)

        output = ttnn.add(identity, projected)
        ttnn.deallocate(projected); ttnn.deallocate(identity)
        return ttnn.reshape(output, (bs, num_tokens, self.embed_dims))

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
