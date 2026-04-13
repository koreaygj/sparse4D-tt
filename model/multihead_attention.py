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

        # Combined QKV weight for self-attention (fused linear)
        w_qkv_pt = torch.cat([parameters["w_q"], parameters["w_k"], parameters["w_v"]], dim=-1)
        b_qkv_pt = torch.cat([parameters["b_q"], parameters["b_k"], parameters["b_v"]], dim=-1)
        self.w_qkv = self._to_device(w_qkv_pt)
        self.b_qkv = self._to_device_bias(b_qkv_pt)

        # Combined QK weight for query=key case (fused Q+K linear, V separate)
        w_qk_pt = torch.cat([parameters["w_q"], parameters["w_k"]], dim=-1)
        b_qk_pt = torch.cat([parameters["b_q"], parameters["b_k"]], dim=-1)
        self.w_qk = self._to_device(w_qk_pt)
        self.b_qk = self._to_device_bias(b_qk_pt)

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
        if query is key and query is value:
            # True self-attention: fused QKV (1 linear instead of 3)
            q_flat = ttnn.reshape(query, (1, 1, bs * num_queries, self.embed_dims))
            qkv = ttnn.linear(q_flat, self.w_qkv, bias=self.b_qkv,
                              compute_kernel_config=self._hifi_compute_config)
            q = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, bs * num_queries, self.embed_dims])
            k = ttnn.slice(qkv, [0, 0, 0, self.embed_dims], [1, 1, bs * num_keys, 2 * self.embed_dims])
            v = ttnn.slice(qkv, [0, 0, 0, 2 * self.embed_dims], [1, 1, bs * num_keys, 3 * self.embed_dims])
            ttnn.deallocate(qkv)
        elif query is key:
            # query=key, value different: fused Q+K (2 linears instead of 3)
            q_flat = ttnn.reshape(query, (1, 1, bs * num_queries, self.embed_dims))
            qk = ttnn.linear(q_flat, self.w_qk, bias=self.b_qk,
                             compute_kernel_config=self._hifi_compute_config)
            q = ttnn.slice(qk, [0, 0, 0, 0], [1, 1, bs * num_queries, self.embed_dims])
            k = ttnn.slice(qk, [0, 0, 0, self.embed_dims], [1, 1, bs * num_keys, 2 * self.embed_dims])
            ttnn.deallocate(qk)
            v_flat = ttnn.reshape(value, (1, 1, bs * num_keys, self.embed_dims))
            v = ttnn.linear(v_flat, self.w_v, bias=self.b_v, compute_kernel_config=self._hifi_compute_config)
        else:
            q_flat = ttnn.reshape(query, (1, 1, bs * num_queries, self.embed_dims))
            q = ttnn.linear(q_flat, self.w_q, bias=self.b_q, compute_kernel_config=self._hifi_compute_config)
            k_flat = ttnn.reshape(key, (1, 1, bs * num_keys, self.embed_dims))
            k = ttnn.linear(k_flat, self.w_k, bias=self.b_k, compute_kernel_config=self._hifi_compute_config)
            v_flat = ttnn.reshape(value, (1, 1, bs * num_keys, self.embed_dims))
            v = ttnn.linear(v_flat, self.w_v, bias=self.b_v, compute_kernel_config=self._hifi_compute_config)

        # Multi-head split: fused reshape+transpose when Q/K same length
        if num_queries == num_keys:
            qkv_cat = ttnn.concat([q, k, v], dim=-1)
            qkv_cat = ttnn.reshape(qkv_cat, (bs, num_queries, 3 * self.embed_dims))
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv_cat, num_heads=self.num_heads, transpose_key=False)
            ttnn.deallocate(qkv_cat)
        else:
            # Cross-attention: pad K/V to match Q length for fused split_heads
            k = ttnn.pad(k, [(0,0),(0,0),(0, num_queries - num_keys),(0,0)], 0.0)
            v = ttnn.pad(v, [(0,0),(0,0),(0, num_queries - num_keys),(0,0)], 0.0)
            qkv_cat = ttnn.concat([q, k, v], dim=-1)
            qkv_cat = ttnn.reshape(qkv_cat, (bs, num_queries, 3 * self.embed_dims))
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                qkv_cat, num_heads=self.num_heads, transpose_key=False)
            ttnn.deallocate(qkv_cat)

        # Scaled dot-product attention via SDPA kernel (FlashAttention-2)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)

        # Reshape back: [b, nh, s, d] → [b, s, nh*d] via concatenate_heads
        attn_output = ttnn.transformer.concatenate_heads(attn_output)
        attn_output = ttnn.reshape(attn_output, (1, 1, bs * num_queries, self.embed_dims))
        output = ttnn.linear(attn_output, self.w_out, bias=self.b_out,
                             compute_kernel_config=self._hifi_compute_config)
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
