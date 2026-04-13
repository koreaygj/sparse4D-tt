# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Sparse4DHead Full Forward (Inference Only)
#
# Connects all sub-modules into the complete decoder pipeline:
#   InstanceBank.get()
#   → anchor_encoder(anchor)
#   → Decoder loop × 6:
#       [temp_gnn] → [gnn] → [norm] → [DFA] → [FFN] → [norm] → [refine]
#   → InstanceBank.update() / cache()
#   → 3D BBox predictions
#
# operation_order (Sparse4D v3, decouple_attn=True):
#   ["deformable", "ffn", "norm", "refine",              ← single-frame (1st)
#    "temp_gnn", "gnn", "norm", "deformable", "ffn", "norm", "refine",  ← multi-frame ×5
#    ...]
# =============================================================================

import torch
import ttnn

from model.multihead_attention import MultiheadAttention, preprocess_mha_parameters
from model.asymmetric_ffn import AsymmetricFFN, preprocess_ffn_parameters
from model.deformable_feature_aggregation import (
    DeformableFeatureAggregation,
    preprocess_dfa_parameters,
)
from model.sparse_box3d_encoder import (
    SparseBox3DEncoder,
    preprocess_encoder_parameters,
)
from model.refinement_module import (
    SparseBox3DRefinementModule,
    preprocess_refinement_parameters,
)
from model.instance_bank import InstanceBank, preprocess_instance_bank_parameters

# Anchor box field indices
X, Y, Z = 0, 1, 2
W, L, H = 3, 4, 5
SIN_YAW, COS_YAW = 6, 7
VX, VY, VZ = 8, 9, 10


class Sparse4DHead:
    """TT-NN Sparse4DHead: full decoder pipeline for inference."""

    def __init__(
        self,
        device,
        parameters: dict,
        operation_order: list,
        embed_dims: int = 256,
        num_decoder: int = 6,
        num_single_frame_decoder: int = 1,
        num_anchor: int = 900,
        num_temp_instances: int = 600,
        num_classes: int = 10,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        num_pts: int = 13,
        num_learnable_pts: int = 6,
        decouple_attn: bool = True,
        spatial_shapes: list = None,
        mesh_device=None,
    ) -> None:
        self.device = mesh_device if mesh_device is not None else device
        self._mesh_device = mesh_device
        self.embed_dims = embed_dims
        self._hifi_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False, packer_l1_acc=False, math_approx_mode=False,
        )
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.num_anchor = num_anchor
        self.num_temp_instances = num_temp_instances
        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.spatial_shapes = spatial_shapes or [(64, 176), (32, 88), (16, 44), (8, 22)]

        mha_embed = embed_dims * 2 if decouple_attn else embed_dims

        # --- Build layers for each operation ---
        self.layers = []
        for i, op in enumerate(operation_order):
            layer_params = parameters["layers"][i]
            # In mesh SPMD mode, DFA uses num_cams=3 (each device processes 3 cameras)
            dfa_num_cams = 3 if mesh_device is not None else num_cams

            if layer_params is None:
                self.layers.append(None)
            elif op in ("gnn", "temp_gnn"):
                self.layers.append(
                    MultiheadAttention(device, layer_params, mha_embed, num_groups, mesh_device=mesh_device)
                )
            elif op == "norm":
                self.layers.append({
                    "weight": self._to_device_1d(layer_params["weight"]),
                    "bias": self._to_device_1d(layer_params["bias"]),
                })
            elif op == "deformable":
                self.layers.append(
                    DeformableFeatureAggregation(
                        device=device,
                        parameters=layer_params,
                        model_config={},
                        embed_dims=embed_dims,
                        num_groups=num_groups,
                        num_levels=num_levels,
                        num_cams=dfa_num_cams,
                        num_pts=num_pts,
                        num_learnable_pts=num_learnable_pts,
                        use_camera_embed=True,
                        residual_mode="cat",
                        mesh_device=mesh_device,
                    )
                )
            elif op == "ffn":
                self.layers.append(
                    AsymmetricFFN(
                        device, layer_params,
                        in_channels=embed_dims * 2,
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims * 4,
                        mesh_device=mesh_device,
                    )
                )
            elif op == "refine":
                self.layers.append(
                    SparseBox3DRefinementModule(
                        device, layer_params,
                        embed_dims=embed_dims,
                        output_dim=11,
                        num_cls=num_classes,
                        refine_yaw=True,
                        with_quality_estimation=True,
                        mesh_device=mesh_device,
                    )
                )

        # --- DFA layer params for mesh DFA setup ---
        self._mesh_dfa_runners = None
        self._dfa_layer_params = {
            i: parameters["layers"][i]
            for i, op in enumerate(operation_order) if op == "deformable"
        }

        # --- Anchor encoder ---
        self.anchor_encoder = SparseBox3DEncoder(
            device, parameters["anchor_encoder"],
            embed_dims=[128, 32, 32, 64] if decouple_attn else embed_dims,
            vel_dims=3,
            mode="cat" if decouple_attn else "add",
            has_output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
            mesh_device=mesh_device,
        )

        # --- Instance bank ---
        self.instance_bank = InstanceBank(
            device=device,
            anchor_data=parameters["instance_bank"]["anchor_data"],
            instance_feature_data=parameters["instance_bank"]["instance_feature_data"],
            num_anchor=num_anchor,
            embed_dims=embed_dims,
            num_temp_instances=num_temp_instances,
            mesh_device=mesh_device,
        )

        # --- Decouple attention fc_before / fc_after ---
        if decouple_attn:
            self.fc_before_weight = self._to_device(parameters["fc_before_weight"])
            self.fc_after_weight = self._to_device(parameters["fc_after_weight"])
        else:
            self.fc_before_weight = None
            self.fc_after_weight = None

    def setup_mesh_dfa(self, dev1, mesh_device=None):
        """Initialize MeshDFARunners for camera-parallel DFA on 2 submeshes."""
        from model.mesh_dfa import MeshDFARunner
        self._mesh_dfa_runners = {}
        for i, op in enumerate(self.operation_order):
            if op == "deformable" and i in self._dfa_layer_params:
                self._mesh_dfa_runners[i] = MeshDFARunner(
                    dfa0=self.layers[i],
                    dev1=dev1,
                    parameters=self._dfa_layer_params[i],
                    model_config={},
                    mesh_device=mesh_device,
                )

    def _to_device(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
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

    def _graph_model(
        self,
        layer_idx: int,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        query_pos: ttnn.Tensor,
        key_pos: ttnn.Tensor,
        bs: int,
        num_queries: int,
        num_keys: int,
    ) -> ttnn.Tensor:
        """Run MHA with decouple_attn wrapping (on device)."""
        if self.decouple_attn:
            query = ttnn.concat([query, query_pos], dim=-1)
            if key is not None and key_pos is not None:
                key = ttnn.concat([key, key_pos], dim=-1)
            else:
                key = query
                num_keys = num_queries
            if value is not None:
                n_val = bs * num_keys
                val_flat = ttnn.reshape(value, (1, 1, n_val, self.embed_dims))
                value = ttnn.linear(val_flat, self.fc_before_weight)
                value = ttnn.reshape(value, (bs, num_keys, self.embed_dims * 2))
        else:
            if query_pos is not None:
                query = ttnn.add(query, query_pos)
            if key is not None and key_pos is not None:
                key = ttnn.add(key, key_pos)
            else:
                key = query
                num_keys = num_queries
            if value is None:
                value = key

        mha = self.layers[layer_idx]
        out = mha.run(query=query, key=key, value=value, bs=bs, num_queries=num_queries, num_keys=num_keys)

        if self.decouple_attn:
            n_q = bs * num_queries
            out_flat = ttnn.reshape(out, (1, 1, n_q, self.embed_dims * 2))
            out = ttnn.linear(out_flat, self.fc_after_weight)
            out = ttnn.reshape(out, (bs, num_queries, self.embed_dims))

        return out

    def _norm(self, layer_idx: int, x: ttnn.Tensor, bs: int, num_tokens: int):
        """LayerNorm on device."""
        norm = self.layers[layer_idx]
        x_flat = ttnn.reshape(x, (1, 1, bs * num_tokens, self.embed_dims))
        normed = ttnn.layer_norm(x_flat, weight=norm["weight"], bias=norm["bias"],
                                  epsilon=1e-5, compute_kernel_config=self._hifi_compute_config)
        return ttnn.reshape(normed, (bs, num_tokens, self.embed_dims))

    def forward(
        self,
        feature_maps: list,
        metas: dict,
        bs: int = 1,
        debug: bool = False,
    ) -> dict:
        """Full Sparse4DHead forward (inference only).

        Args:
            feature_maps: list of ttnn.Tensor from FPN [1, 1, N*H*W, C]
            metas: dict with 'timestamp', 'img_metas', 'projection_mat', 'image_wh'
            bs: batch size
            debug: if True, collect intermediate tensors for PCC comparison

        Returns:
            dict with 'classification', 'prediction', 'quality', and optionally 'debug'
        """
        num_anchor = self.num_anchor
        debug_intermediates = {}

        # Clear per-frame DFA caches
        for i, op in enumerate(self.operation_order):
            if op == "deformable" and self.layers[i] is not None:
                self.layers[i]._cached_camera_embed = None
                self.layers[i]._host_proj = None
                self.layers[i]._host_wh = None
                self.layers[i]._cached_proj_rm = None  # proj changes per frame

        # 1. InstanceBank.get()
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(bs, metas)

        # 2. Anchor encoding
        anchor_embed = self.anchor_encoder.run(anchor, bs=bs, num_anchor=num_anchor)

        if debug:
            debug_intermediates["anchor_embed_init"] = ttnn.to_torch(anchor_embed)
            debug_intermediates["instance_feature_init"] = ttnn.to_torch(instance_feature)

        if temp_anchor is not None:
            num_temp = self.num_temp_instances
            temp_anchor_embed = self.anchor_encoder.run(
                temp_anchor, bs=bs, num_anchor=num_temp
            )
        else:
            temp_anchor_embed = None
            num_temp = 0

        # 3. Pre-allocate projection_mat and image_wh on device
        proj_pt = metas["projection_mat"].float()
        if self._mesh_device is not None:
            proj_tt = ttnn.from_torch(
                proj_pt,
                layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensorToMesh(self._mesh_device, dim=1),
            )
            # image_wh is constant across frames — cache on device
            if not hasattr(self, '_cached_wh_tt') or self._cached_wh_tt is None:
                wh_pt = metas["image_wh"].float()
                self._cached_wh_tt = ttnn.from_torch(
                    wh_pt,
                    layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16,
                    mesh_mapper=ttnn.ShardTensorToMesh(self._mesh_device, dim=1),
                )
            wh_tt = self._cached_wh_tt
        else:
            proj_tt = ttnn.from_torch(
                proj_pt, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16,
            )
            if not hasattr(self, '_cached_wh_tt') or self._cached_wh_tt is None:
                wh_pt = metas["image_wh"].float()
                self._cached_wh_tt = ttnn.from_torch(
                    wh_pt, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16,
                )
            wh_tt = self._cached_wh_tt

        # 4. Decoder loop
        prediction = []
        classification = []
        quality = []

        for i, op in enumerate(self.operation_order):

            if self.layers[i] is None:
                continue

            elif op == "temp_gnn":
                if temp_instance_feature is not None:
                    instance_feature = self._graph_model(
                        i,
                        query=instance_feature,
                        key=temp_instance_feature,
                        value=temp_instance_feature,
                        query_pos=anchor_embed,
                        key_pos=temp_anchor_embed,
                        bs=bs,
                        num_queries=num_anchor,
                        num_keys=num_temp,
                    )
                else:
                    # No temporal data: self-attention fallback
                    # PyTorch passes key=None, value=None → MHA uses query as K and V
                    instance_feature = self._graph_model(
                        i,
                        query=instance_feature,
                        key=None,
                        value=None,
                        query_pos=anchor_embed,
                        key_pos=None,
                        bs=bs,
                        num_queries=num_anchor,
                        num_keys=num_anchor,
                    )

            elif op == "gnn":
                instance_feature = self._graph_model(
                    i,
                    query=instance_feature,
                    key=None,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    key_pos=None,
                    bs=bs,
                    num_queries=num_anchor,
                    num_keys=num_anchor,
                )

            elif op == "norm":
                instance_feature = self._norm(i, instance_feature, bs, num_anchor)

            elif op == "deformable":
                dfa = self.layers[i]
                instance_feature = dfa.run(
                    instance_feature, anchor, anchor_embed,
                    feature_maps, proj_tt, wh_tt,
                    self.spatial_shapes, bs=bs, num_anchor=num_anchor,
                )

            elif op == "ffn":
                ffn = self.layers[i]
                instance_feature = ffn.run(
                    instance_feature, bs=bs, num_tokens=num_anchor
                )

            elif op == "refine":
                refine = self.layers[i]
                return_cls = (
                    len(prediction) == self.num_single_frame_decoder - 1
                    or i == len(self.operation_order) - 1
                )
                anchor, cls, qt = refine.run(
                    instance_feature, anchor, anchor_embed, time_interval,
                    bs=bs, num_anchor=num_anchor, return_cls=return_cls,
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)

                # After single-frame decoder: update instances
                if len(prediction) == self.num_single_frame_decoder:
                    if cls is not None:
                        old_instance_feature = instance_feature
                        instance_feature, anchor = self.instance_bank.update(
                            instance_feature, anchor, cls, bs=bs,
                        )
                        # Only deallocate if update created a new tensor
                        if instance_feature is not old_instance_feature:
                            ttnn.deallocate(old_instance_feature)

                # Re-encode anchor (except at last step)
                if i != len(self.operation_order) - 1:
                    old_anchor_embed = anchor_embed
                    anchor_embed = self.anchor_encoder.run(
                        anchor, bs=bs, num_anchor=num_anchor
                    )
                    ttnn.deallocate(old_anchor_embed)

                # Update temp_anchor_embed after single-frame decoder
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = ttnn.slice(
                        anchor_embed,
                        [0, 0, 0],
                        [bs, self.num_temp_instances, self.embed_dims],
                    )

            if debug:
                debug_intermediates[f"step_{i}_{op}"] = ttnn.to_torch(instance_feature)
                if op == "refine":
                    debug_intermediates[f"step_{i}_anchor"] = ttnn.to_torch(anchor)

        # 5. Deallocate projection_mat and image_wh
        ttnn.deallocate(proj_tt)
        # wh_tt is cached, don't deallocate

        # 6. Cache for next frame
        last_cls = classification[-1]
        if last_cls is not None:
            self.instance_bank.cache(
                instance_feature, anchor, last_cls, metas, bs=bs,
            )

        result = {
            "prediction": prediction,
            "classification": classification,
            "quality": quality,
        }
        if debug:
            result["debug"] = debug_intermediates
        return result


def preprocess_sparse4d_head_parameters(pt_head) -> dict:
    """Extract all parameters from PyTorch Sparse4DHead.

    Args:
        pt_head: PyTorch Sparse4DHead instance (from mmdet3d)

    Returns:
        dict with all sub-module parameters
    """
    params = {}

    # Layers (per operation_order)
    layer_params = []
    for i, op in enumerate(pt_head.operation_order):
        layer = pt_head.layers[i]
        if layer is None:
            layer_params.append(None)
        elif op in ("gnn", "temp_gnn"):
            layer_params.append(preprocess_mha_parameters(layer))
        elif op == "norm":
            layer_params.append({
                "weight": layer.weight.data.clone(),
                "bias": layer.bias.data.clone(),
            })
        elif op == "deformable":
            layer_params.append(preprocess_dfa_parameters(layer))
        elif op == "ffn":
            layer_params.append(preprocess_ffn_parameters(layer))
        elif op == "refine":
            layer_params.append(preprocess_refinement_parameters(layer))
    params["layers"] = layer_params

    # Anchor encoder
    params["anchor_encoder"] = preprocess_encoder_parameters(pt_head.anchor_encoder)

    # Instance bank
    params["instance_bank"] = preprocess_instance_bank_parameters(pt_head.instance_bank)

    # Decouple attention fc
    if pt_head.decouple_attn:
        params["fc_before_weight"] = pt_head.fc_before.weight.data.clone().t()
        params["fc_after_weight"] = pt_head.fc_after.weight.data.clone().t()

    return params
