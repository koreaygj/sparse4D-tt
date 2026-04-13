# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# DeformableFeatureAggregation for TT Devices
#
# Forward flow:
#   1. kps_generator: anchor → 3D key points
#      - fixed/learnable key points: device (ttnn ops)
#      - rotation + translation: HOST (ttnn.slice hang workaround, small tensor ~35K elements)
#   2. project_points: 3D → 2D via projection matrix (ttnn.matmul, device)
#   3. get_weights: instance_feature → attention weights (ttnn.linear + softmax, device)
#   4. feature_sampling: grid_sample per FPN level (ttnn.grid_sample_lerp, device)
#      - lerp-based bilinear: 2-pass lerp reduces BF16 rounding from 2x to 1x
#   5. multi_view_level_fusion: weighted sum (ttnn.multiply + ttnn.sum, device)
#   6. output_proj: ttnn.linear + residual (device)
# =============================================================================

from typing import Dict, List, Tuple

import torch
import ttnn

# Anchor box field indices (Sparse4D convention)
X, Y, Z = 0, 1, 2
W, L, H = 3, 4, 5
SIN_YAW, COS_YAW = 6, 7
VX, VY, VZ = 8, 9, 10


class DeformableFeatureAggregation:
    def __init__(
        self,
        device,
        parameters,
        model_config: Dict,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        num_pts: int = 13,
        num_learnable_pts: int = 6,
        use_camera_embed: bool = True,
        residual_mode: str = "cat",
        mesh_device=None,
    ) -> None:
        self.device = mesh_device if mesh_device is not None else device
        self._mesh_device = mesh_device
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = embed_dims // num_groups  # 32
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.num_pts = num_pts
        self.num_learnable_pts = num_learnable_pts
        self.use_camera_embed = use_camera_embed
        self.residual_mode = residual_mode
        self.model_config = model_config

        # HiFi2 compute config
        self._hifi_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )

        # Weight CLP reorder index: camera-major → level-major
        # Needed because transposed grid produces level-major feature ordering
        total_clp = num_cams * num_levels * num_pts  # 156
        perm = []
        for l in range(num_levels):
            for c in range(num_cams):
                for p in range(num_pts):
                    perm.append(c * num_levels * num_pts + l * num_pts + p)
        _perm_t = torch.tensor(perm, dtype=torch.int32).reshape(1, total_clp, 1)
        _perm_t = _perm_t.expand(900, total_clp, num_groups).contiguous()
        _perm_kw = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.uint32)
        if self._mesh_device is not None:
            _perm_kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        self._wt_perm_idx = ttnn.from_torch(_perm_t, **_perm_kw)

        # L1 sharded grid config for grid_sample (PR #28308)
        # kps_project_fused output: [nc, N, 1, 32] (padded from 26, L1-aligned)
        # total sticks = nc*N = 2700, K = 32/2 = 16 (13 real + 3 padding zeros)
        _total_sticks = num_cams * 900  # 2700
        _shard_h = (_total_sticks + 55) // 56  # 49
        _padded_last = ((num_pts * 2 + 7) // 8) * 8  # 26 → 32
        _core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}
        )
        _shard_spec = ttnn.ShardSpec(
            _core_grid, (_shard_h, _padded_last), ttnn.ShardOrientation.ROW_MAJOR
        )
        self._grid_sharded_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec
        )

        # L1 sharded config for precomputed grid (pts*6 per point instead of pts*2)
        _padded_precomputed = ((num_pts * 6 + 7) // 8) * 8  # 78 → 80
        _shard_spec_precomputed = ttnn.ShardSpec(
            _core_grid, (_shard_h, _padded_precomputed), ttnn.ShardOrientation.ROW_MAJOR
        )
        self._grid_precomputed_sharded_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec_precomputed
        )

        # Pre-allocate scalar constants on device (reused per camera in _project_points)
        _skw = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            _skw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        self._scalar_half = ttnn.from_torch(torch.full((1, 1, 1), 0.5), **_skw)

        # --- Move all parameters to TT device ---
        # Note: PyTorch nn.Linear stores weight as [out, in],
        # but ttnn.linear expects weight as [in, out], so we transpose.

        # KPS Generator
        self.fix_scale = self._to_device(parameters["kps_fix_scale"])  # [7, 3]
        self.learnable_fc_weight = self._to_device(
            parameters["kps_learnable_fc_weight"].t()
        )  # [256, 18]
        self.learnable_fc_bias = self._to_device_bias(
            parameters["kps_learnable_fc_bias"]
        )  # [1,1,1,18]

        # Camera encoder
        if use_camera_embed:
            self.cam_linear1_weight = self._to_device(
                parameters["cam_linear1_weight"].t()
            )  # [12, 256]
            self.cam_linear1_bias = self._to_device_bias(parameters["cam_linear1_bias"])
            self.cam_ln1_weight = self._to_device_1d(parameters["cam_ln1_weight"])
            self.cam_ln1_bias = self._to_device_1d(parameters["cam_ln1_bias"])
            self.cam_linear2_weight = self._to_device(
                parameters["cam_linear2_weight"].t()
            )  # [256, 256]
            self.cam_linear2_bias = self._to_device_bias(parameters["cam_linear2_bias"])
            self.cam_ln2_weight = self._to_device_1d(parameters["cam_ln2_weight"])
            self.cam_ln2_bias = self._to_device_1d(parameters["cam_ln2_bias"])

        # Weights FC
        self.weights_fc_weight = self._to_device(
            parameters["weights_fc_weight"].t()
        )  # [256, 416]
        self.weights_fc_bias = self._to_device_bias(
            parameters["weights_fc_bias"]
        )  # [1,1,1,416]

        # Output projection
        self.output_proj_weight = self._to_device(
            parameters["output_proj_weight"].t()
        )  # [256, 256]
        self.output_proj_bias = self._to_device_bias(
            parameters["output_proj_bias"]
        )  # [1,1,1,256]

        # CCL helper for mesh combine
        if self._mesh_device is not None:
            from models.common.modules.tt_ccl import TT_CCL

            self._tt_ccl = TT_CCL(self._mesh_device)

    def _to_device(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Move weight tensor to device in TILE layout."""
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

    def _to_device_bias(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Move bias tensor to device as [1, 1, 1, N] in TILE layout."""
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

    def _to_device_1d(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Move 1D tensor (LayerNorm weight/bias) to device as [1, 1, 1, N]."""
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

    def _kps_generator_pre_rotation(self, anchor, instance_feature, bs, num_anchor):
        """Generate pre-rotation 3D key points for kps_project_fused kernel."""
        n = bs * num_anchor
        size_wlh = ttnn.slice(anchor, [0, 0, W], [bs, num_anchor, H + 1])
        size = ttnn.exp(size_wlh)
        ttnn.deallocate(size_wlh)
        size_3d = ttnn.reshape(size, (n, 1, 3))
        fix_scale_3d = ttnn.reshape(self.fix_scale, (1, 7, 3))
        fix_kps = ttnn.multiply(fix_scale_3d, size_3d)
        inst_flat = ttnn.reshape(instance_feature, (1, 1, n, self.embed_dims))
        learnable = ttnn.linear(
            inst_flat,
            self.learnable_fc_weight,
            bias=self.learnable_fc_bias,
            compute_kernel_config=self._hifi_compute_config,
        )
        learnable = ttnn.reshape(learnable, (n, self.num_learnable_pts, 3))
        learnable = ttnn.sigmoid(learnable)
        learnable = ttnn.subtract(learnable, self._scalar_half)
        learnable_kps = ttnn.multiply(learnable, size_3d)
        ttnn.deallocate(learnable)
        key_points = ttnn.concat([fix_kps, learnable_kps], dim=1)
        ttnn.deallocate(fix_kps)
        ttnn.deallocate(learnable_kps)
        key_points = ttnn.to_layout(key_points, ttnn.ROW_MAJOR_LAYOUT)
        key_points = ttnn.to_memory_config(key_points, ttnn.DRAM_MEMORY_CONFIG)
        return key_points

    def _camera_encoder(
        self,
        projection_mat: ttnn.Tensor,
        bs: int,
    ) -> ttnn.Tensor:
        """Camera encoder on device: Linear→ReLU→LN→Linear→ReLU→LN.

        Args:
            projection_mat: [bs, num_cams, 4, 4] on device

        Returns:
            camera_embed: [bs, num_cams, 256] on device (TILE)
        """
        # Extract first 3 rows of 4x4: [bs, num_cams, 3, 4] -> [bs, num_cams, 12]
        cam_input = ttnn.slice(projection_mat, [0, 0, 0, 0], [bs, self.num_cams, 3, 4])
        cam_input = ttnn.reshape(cam_input, (1, 1, bs * self.num_cams, 12))

        # Linear1+ReLU fused: [bs*num_cams, 12] -> [bs*num_cams, 256]
        x = ttnn.linear(cam_input, self.cam_linear1_weight, bias=self.cam_linear1_bias,
                         activation="relu", compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(cam_input)
        ln_in = x
        x = ttnn.layer_norm(x, weight=self.cam_ln1_weight, bias=self.cam_ln1_bias,
                             epsilon=1e-5, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(ln_in)

        # Linear2: [bs*num_cams, 256] -> [bs*num_cams, 256]
        linear2_in = x
        x = ttnn.linear(x, self.cam_linear2_weight, bias=self.cam_linear2_bias,
                         activation="relu", compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(linear2_in)
        ln_in = x
        x = ttnn.layer_norm(x, weight=self.cam_ln2_weight, bias=self.cam_ln2_bias,
                             epsilon=1e-5, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(ln_in)

        # Reshape to [bs, num_cams, 256]
        x = ttnn.reshape(x, (bs, self.num_cams, self.embed_dims))
        return x

    def _get_weights(
        self,
        instance_feature: ttnn.Tensor,
        anchor_embed: ttnn.Tensor,
        projection_mat: ttnn.Tensor,
        bs: int,
        num_anchor: int,
        return_logits: bool = False,
    ) -> ttnn.Tensor:
        """Compute attention weights on device.

        Args:
            instance_feature: [bs, num_anchor, embed_dims] on device
            anchor_embed: [bs, num_anchor, embed_dims] on device
            projection_mat: [bs, num_cams, 4, 4] on device

        Returns:
            weights: [bs*num_anchor, num_cams*num_levels*num_pts, num_groups] on device
        """
        feature = ttnn.add(instance_feature, anchor_embed)  # [bs, num_anchor, 256]

        if self.use_camera_embed:
            # Cache camera_embed per frame (same projection_mat across all 6 DFA calls)
            if (
                not hasattr(self, "_cached_camera_embed")
                or self._cached_camera_embed is None
            ):
                self._cached_camera_embed = self._camera_encoder(projection_mat, bs)
            camera_embed = self._cached_camera_embed
            feat_exp = ttnn.reshape(feature, (bs, num_anchor, 1, self.embed_dims))
            cam_exp = ttnn.reshape(
                camera_embed, (bs, 1, self.num_cams, self.embed_dims)
            )
            feature = ttnn.add(feat_exp, cam_exp)
            # Don't deallocate camera_embed — it's cached for reuse
            feature = ttnn.reshape(
                feature, (1, 1, bs * num_anchor * self.num_cams, self.embed_dims)
            )
        else:
            feature = ttnn.reshape(feature, (1, 1, bs * num_anchor, self.embed_dims))

        weights = ttnn.linear(
            feature,
            self.weights_fc_weight,
            bias=self.weights_fc_bias,
            compute_kernel_config=self._hifi_compute_config,
        )

        ttnn.deallocate(feature)

        total_clp = self.num_cams * self.num_levels * self.num_pts
        if self.use_camera_embed:
            weights = ttnn.reshape(
                weights,
                (
                    1,
                    1,
                    bs * num_anchor,
                    self.num_cams * self.num_levels * self.num_pts * self.num_groups,
                ),
            )
            weights = ttnn.reshape(
                weights, (bs * num_anchor, total_clp, self.num_groups)
            )
        else:
            weights = ttnn.reshape(
                weights, (bs * num_anchor, total_clp, self.num_groups)
            )

        if return_logits:
            return weights  # pre-softmax logits

        weights = ttnn.softmax(
            weights,
            dim=1,
            numeric_stable=True,
            compute_kernel_config=self._hifi_compute_config,
        )

        return weights

    def run(
        self,
        instance_feature: ttnn.Tensor,
        anchor: ttnn.Tensor,
        anchor_embed: ttnn.Tensor,
        feature_maps: List[ttnn.Tensor],
        projection_mat: ttnn.Tensor,
        image_wh: ttnn.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        bs: int,
        num_anchor: int,
    ) -> ttnn.Tensor:
        n = bs * num_anchor
        nc = self.num_cams

        # 1. Start attention weights early (independent of KPS projection)
        weights = self._get_weights(
            instance_feature, anchor_embed, projection_mat, bs, num_anchor
        )

        # 2. Pre-rotation key points + fused KPS projection (overlaps with weight compute on device)
        key_points = self._kps_generator_pre_rotation(
            anchor, instance_feature, bs, num_anchor
        )
        anchor_rm = ttnn.reshape(anchor, (n, 1, 11))
        anchor_rm = ttnn.to_layout(anchor_rm, ttnn.ROW_MAJOR_LAYOUT)

        if not hasattr(self, "_cached_proj_rm") or self._cached_proj_rm is None:
            proj_flat = ttnn.reshape(projection_mat, (nc * 4, 4))
            proj_flat = ttnn.to_layout(proj_flat, ttnn.ROW_MAJOR_LAYOUT)
            proj_flat = ttnn.to_memory_config(proj_flat, ttnn.DRAM_MEMORY_CONFIG)
            proj_padded = ttnn.pad(proj_flat, [(0, 0), (0, 28)], 0.0)
            proj_padded = ttnn.typecast(proj_padded, ttnn.float32)
            self._cached_proj_rm = ttnn.slice(proj_padded, [0, 0], [nc * 4, 4])
            ttnn.deallocate(proj_padded)
            self._cached_proj_rm = ttnn.reshape(self._cached_proj_rm, (nc, 4, 4))
            self._cached_proj_rm = ttnn.to_memory_config(
                self._cached_proj_rm, ttnn.DRAM_MEMORY_CONFIG
            )

        if not hasattr(self, "_cached_wh_rm") or self._cached_wh_rm is None:
            wh_flat = ttnn.reshape(image_wh, (nc, 2))
            wh_flat = ttnn.to_layout(wh_flat, ttnn.ROW_MAJOR_LAYOUT)
            wh_flat = ttnn.to_memory_config(wh_flat, ttnn.DRAM_MEMORY_CONFIG)
            wh_padded = ttnn.pad(wh_flat, [(0, 0), (0, 30)], 0.0)
            wh_padded = ttnn.typecast(wh_padded, ttnn.float32)
            self._cached_wh_rm = ttnn.slice(wh_padded, [0, 0], [nc, 2])
            ttnn.deallocate(wh_padded)
            self._cached_wh_rm = ttnn.reshape(self._cached_wh_rm, (nc, 1, 2))
            self._cached_wh_rm = ttnn.to_memory_config(
                self._cached_wh_rm, ttnn.DRAM_MEMORY_CONFIG
            )

        points_2d = ttnn.kps_project_fused(
            key_points,
            anchor_rm,
            self._cached_proj_rm,
            self._cached_wh_rm,
            num_cams=nc,
            num_pts=self.num_pts,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            precompute_grid=True,
            spatial_shapes=[(h, w) for h, w in spatial_shapes],
        )
        ttnn.deallocate(key_points)
        ttnn.deallocate(anchor_rm)

        # Precomputed output: [nc*NL, N, 1, pts*6] camera-major
        # Layout: [cam0_L0, cam0_L1, cam0_L2, cam0_L3, cam1_L0, ...]
        pts6 = points_2d.shape[-1]

        # 3. Start weight transpose early (overlaps with grid_sample on device)
        weights_t = ttnn.transpose(weights, 0, 1)

        # 4. Grid sample with precomputed grids + transposed s2i
        total_clp = nc * self.num_levels * self.num_pts
        if not hasattr(self, '_rearrange_buf') or self._rearrange_buf is None:
            _buf = torch.zeros(total_clp, n, self.embed_dims, dtype=torch.bfloat16)
            _kw = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            if self._mesh_device is not None:
                _kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
            self._rearrange_buf = ttnn.from_torch(_buf, **_kw)

        for level_idx, fm_tt in enumerate(feature_maps):
            # Extract per-level precomputed grid from camera-major output
            # Camera-major: row c*NL+l = camera c, level l
            # Gather per-camera slices and concat to get [nc, N, 1, pts*6]
            cam_grids = []
            for c in range(nc):
                row = c * self.num_levels + level_idx
                cam_grid = ttnn.slice(
                    points_2d,
                    [row, 0, 0, 0],
                    [row + 1, n, 1, pts6],
                )
                cam_grids.append(cam_grid)
            level_grid = ttnn.concat(cam_grids, dim=0)
            for cg in cam_grids:
                ttnn.deallocate(cg)
            # L1 shard the precomputed grid
            level_grid_sh = ttnn.to_memory_config(level_grid, self._grid_precomputed_sharded_mem)
            ttnn.deallocate(level_grid)
            sampled = ttnn.grid_sample(
                fm_tt, level_grid_sh, padding_mode="zeros", align_corners=False,
                use_precomputed_grid=True,
            )
            ttnn.deallocate(level_grid_sh)
            # Transposed s2i: write L1 sharded → DRAM [CLP, N, C] in camera-major order
            ttnn.transposed_s2i(
                sampled, self._rearrange_buf,
                num_cams=nc, num_pts=self.num_pts, num_anchors=n,
                num_levels=self.num_levels, level=level_idx,
            )
        ttnn.deallocate(points_2d)

        # 5. Tilize + GWS
        features = ttnn.to_layout(self._rearrange_buf, ttnn.TILE_LAYOUT)
        gws_out = ttnn.grouped_weighted_sum(
            features, weights_t, num_groups=self.num_groups, group_dims=self.group_dims
        )
        ttnn.deallocate(weights_t)
        ttnn.deallocate(features)
        n_padded = ((n + 31) // 32) * 32
        chunk0 = ttnn.slice(gws_out, [0, 0], [n, self.embed_dims])
        chunk1 = ttnn.slice(gws_out, [n_padded, 0], [n_padded + n, self.embed_dims])
        ttnn.deallocate(gws_out)
        features = ttnn.add(chunk0, chunk1)
        ttnn.deallocate(chunk0)
        ttnn.deallocate(chunk1)
        features = ttnn.reshape(features, (1, 1, n, self.embed_dims))

        # 6. Mesh combine via CCL (features already in DRAM from add)
        if self._mesh_device is not None:
            features = ttnn.all_reduce(
                features,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )

        # 7. Output projection + residual
        output = ttnn.linear(
            features,
            self.output_proj_weight,
            bias=self.output_proj_bias,
            compute_kernel_config=self._hifi_compute_config,
        )
        ttnn.deallocate(features)
        inst_flat = ttnn.reshape(instance_feature, (1, 1, n, self.embed_dims))
        if output.dtype != inst_flat.dtype:
            output = ttnn.typecast(output, inst_flat.dtype)
        if self.residual_mode == "add":
            output = ttnn.add(output, inst_flat)
            output = ttnn.reshape(output, (bs, num_anchor, self.embed_dims))
        elif self.residual_mode == "cat":
            output = ttnn.concat([output, inst_flat], dim=-1)
            output = ttnn.reshape(output, (bs, num_anchor, 2 * self.embed_dims))

        return output


def preprocess_dfa_parameters(pt_model) -> dict:
    """Extract parameters from PyTorch DeformableFeatureAggregation model.

    Args:
        pt_model: PyTorch DeformableFeatureAggregation instance
            (from mmdet3d_plugin.models.blocks)

    Returns:
        dict of torch tensors for DeformableFeatureAggregation.__init__
    """
    params = {}

    # KPS Generator
    params["kps_fix_scale"] = pt_model.kps_generator.fix_scale.data.clone()
    params["kps_learnable_fc_weight"] = (
        pt_model.kps_generator.learnable_fc.weight.data.clone()
    )
    params["kps_learnable_fc_bias"] = (
        pt_model.kps_generator.learnable_fc.bias.data.clone()
    )

    # Camera encoder (if exists)
    if pt_model.camera_encoder is not None:
        enc = pt_model.camera_encoder
        params["cam_linear1_weight"] = enc[0].weight.data.clone()
        params["cam_linear1_bias"] = enc[0].bias.data.clone()
        params["cam_ln1_weight"] = enc[2].weight.data.clone()
        params["cam_ln1_bias"] = enc[2].bias.data.clone()
        params["cam_linear2_weight"] = enc[3].weight.data.clone()
        params["cam_linear2_bias"] = enc[3].bias.data.clone()
        params["cam_ln2_weight"] = enc[5].weight.data.clone()
        params["cam_ln2_bias"] = enc[5].bias.data.clone()

    # Weights FC
    params["weights_fc_weight"] = pt_model.weights_fc.weight.data.clone()
    params["weights_fc_bias"] = pt_model.weights_fc.bias.data.clone()

    # Output projection
    params["output_proj_weight"] = pt_model.output_proj.weight.data.clone()
    params["output_proj_bias"] = pt_model.output_proj.bias.data.clone()

    return params
