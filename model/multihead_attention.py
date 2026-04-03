# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# MultiheadAttention for TT Devices
#
# Implements mmcv's MultiheadAttention wrapper around nn.MultiheadAttention
# using ttnn ops. Used as "gnn" (self-attention) and "temp_gnn" (cross-attention)
# in the Sparse4D decoder.
#
# Forward flow:
#   1. Q/K/V linear projections (from in_proj_weight/bias)
#   2. Multi-head reshape + transpose
#   3. Scaled dot-product attention: softmax(QK^T / sqrt(d)) * V
#   4. Output projection (out_proj)
#   5. Residual connection: identity + output
#
# In Sparse4D with decouple_attn=True:
#   - query = cat([instance_feature, anchor_embed], dim=-1)  [bs, N, 512]
#   - key = cat([key, key_pos], dim=-1) or query             [bs, M, 512]
#   - value = fc_before(value)                                [bs, M, 512]
#   - output = fc_after(MHA(query, key, value))               [bs, N, 256]
# =============================================================================

import torch
import ttnn
from loguru import logger


class MultiheadAttention:
    """TT-NN implementation of multi-head attention.

    Mirrors mmcv.cnn.bricks.transformer.MultiheadAttention behavior:
    - Wraps scaled dot-product attention with Q/K/V projections
    - Supports both self-attention (key=None) and cross-attention
    - Includes residual connection (identity + output)
    - Dropout is skipped (inference only)
    """

    def __init__(
        self,
        device,
        parameters: dict,
        embed_dims: int = 512,
        num_heads: int = 8,
        mesh_device=None,
    ) -> None:
        self.device = mesh_device if mesh_device is not None else device
        self._mesh_device = mesh_device
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self._hifi_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False, packer_l1_acc=False, math_approx_mode=False,
        )

        # Q, K, V projection weights (already transposed: [in, out] for ttnn.linear)
        self.w_q = self._to_device(parameters["w_q"])
        self.b_q = self._to_device_bias(parameters["b_q"])
        self.w_k = self._to_device(parameters["w_k"])
        self.b_k = self._to_device_bias(parameters["b_k"])
        self.w_v = self._to_device(parameters["w_v"])
        self.b_v = self._to_device_bias(parameters["b_v"])

        # Output projection
        self.w_out = self._to_device(parameters["w_out"])
        self.b_out = self._to_device_bias(parameters["b_out"])

        # Precompute scale factor on device
        self.scale = self._to_device_bias(torch.full((1,), self.head_dim**-0.5))

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

    def run(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        bs: int,
        num_queries: int,
        num_keys: int,
    ) -> ttnn.Tensor:
        """Forward pass of multi-head attention.

        Implements: identity + out_proj(softmax(QK^T/sqrt(d)) @ V)
        where identity = query (residual connection).

        Args:
            query: [bs, num_queries, embed_dims] on device (TILE)
            key: [bs, num_keys, embed_dims] on device (TILE)
            value: [bs, num_keys, embed_dims] on device (TILE)
            bs: batch size
            num_queries: number of query tokens
            num_keys: number of key/value tokens

        Returns:
            output: [bs, num_queries, embed_dims] on device (TILE)
        """
        # Save identity for residual
        identity = query

        # Handle key=None, value=None (self-attention: use query for all)
        if key is None:
            key = query
            num_keys = num_queries
        if value is None:
            value = query

        # --- 1. Linear projections ---
        logger.debug("MHA: Q reshape start")
        q_flat = ttnn.reshape(query, (1, 1, bs * num_queries, self.embed_dims))
        logger.debug(f"MHA: Q linear start, q={q_flat.shape}")
        q = ttnn.linear(q_flat, self.w_q, bias=self.b_q, compute_kernel_config=self._hifi_compute_config)
        k_flat = ttnn.reshape(key, (1, 1, bs * num_keys, self.embed_dims))
        k = ttnn.linear(k_flat, self.w_k, bias=self.b_k, compute_kernel_config=self._hifi_compute_config)
        v_flat = ttnn.reshape(value, (1, 1, bs * num_keys, self.embed_dims))
        v = ttnn.linear(v_flat, self.w_v, bias=self.b_v, compute_kernel_config=self._hifi_compute_config)

        # Multi-head reshape + permute
        q = ttnn.reshape(q, (bs, num_queries, self.num_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (bs, num_keys, self.num_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (bs, num_keys, self.num_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        k_t = ttnn.transpose(k, -2, -1)
        attn_weights = ttnn.matmul(q, k_t, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(k_t)

        attn_weights = ttnn.multiply(attn_weights, self.scale)
        attn_weights = ttnn.softmax(attn_weights, dim=-1, numeric_stable=True,
                                    compute_kernel_config=self._hifi_compute_config)

        attn_output = ttnn.matmul(attn_weights, v, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(attn_weights); ttnn.deallocate(v)

        # Reshape back + output projection
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (1, 1, bs * num_queries, self.embed_dims))
        output = ttnn.linear(attn_output, self.w_out, bias=self.b_out, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(attn_output)

        # --- 6. Residual connection ---
        identity_flat = ttnn.reshape(
            identity, (1, 1, bs * num_queries, self.embed_dims)
        )
        output = ttnn.add(output, identity_flat)

        output = ttnn.reshape(output, (bs, num_queries, self.embed_dims))

        return output


def preprocess_mha_parameters(pt_mha_layer) -> dict:
    """Extract parameters from mmcv MultiheadAttention.

    mmcv's MultiheadAttention wraps nn.MultiheadAttention which stores:
    - attn.in_proj_weight: [3*embed_dims, embed_dims]
    - attn.in_proj_bias: [3*embed_dims]
    - attn.out_proj.weight: [embed_dims, embed_dims]
    - attn.out_proj.bias: [embed_dims]

    We split in_proj into separate Q, K, V weights and transpose
    for ttnn.linear (expects [in_features, out_features]).

    Args:
        pt_mha_layer: mmcv MultiheadAttention instance

    Returns:
        dict of torch tensors
    """
    params = {}
    attn = pt_mha_layer.attn

    # Split in_proj_weight [3*E, E] into Q, K, V each [E, E]
    in_proj_weight = attn.in_proj_weight.data.clone()
    w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
    # Transpose: nn.Linear [out, in] -> ttnn.linear [in, out]
    params["w_q"] = w_q.t()
    params["w_k"] = w_k.t()
    params["w_v"] = w_v.t()

    # Split in_proj_bias [3*E] into Q, K, V
    in_proj_bias = attn.in_proj_bias.data.clone()
    b_q, b_k, b_v = in_proj_bias.chunk(3, dim=0)
    params["b_q"] = b_q
    params["b_k"] = b_k
    params["b_v"] = b_v

    # Output projection
    params["w_out"] = attn.out_proj.weight.data.clone().t()
    params["b_out"] = attn.out_proj.bias.data.clone()

    return params
