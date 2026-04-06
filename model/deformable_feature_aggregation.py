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
from loguru import logger

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

        # HiFi compute config for precision-sensitive operations
        self._hifi_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        # Pre-allocate scalar constants on device (reused per camera in _project_points)
        _skw = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            _skw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        self._scalar_two = ttnn.from_torch(torch.full((1, 1, 1), 2.0), **_skw)
        self._scalar_one = ttnn.from_torch(torch.full((1, 1, 1), 1.0), **_skw)
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

    def _kps_generator(
        self,
        anchor: ttnn.Tensor,
        instance_feature: ttnn.Tensor,
        bs: int,
        num_anchor: int,
    ) -> ttnn.Tensor:
        """Generate 3D key points from anchor boxes on device.

        Args:
            anchor: [bs, num_anchor, 11] on device (TILE)
            instance_feature: [bs, num_anchor, embed_dims] on device (TILE)

        Returns:
            key_points: [bs*num_anchor*num_pts, 3] on device
        """
        # Extract size [W, L, H] and compute exp
        # anchor [..., 3:6] -> size
        logger.debug(f"_kps_generator: slice+exp, anchor shape={anchor.shape}")
        size_wlh = ttnn.slice(
            anchor, [0, 0, W], [bs, num_anchor, H + 1]
        )  # [bs, num_anchor, 3]
        size = ttnn.exp(size_wlh)  # [bs, num_anchor, 3]
        ttnn.deallocate(size_wlh)
        logger.debug("_kps_generator: slice+exp done")

        # Reshape size for broadcasting: [bs*num_anchor, 1, 3]
        size_3d = ttnn.reshape(size, (bs * num_anchor, 1, 3))

        # Fixed key points: fix_scale [7, 3] * size [bs*num_anchor, 1, 3]
        fix_scale_3d = ttnn.reshape(self.fix_scale, (1, 7, 3))
        fix_kps = ttnn.multiply(fix_scale_3d, size_3d)  # [bs*num_anchor, 7, 3]
        logger.debug("_kps_generator: fix_kps done")

        # Learnable key points
        inst_flat = ttnn.reshape(
            instance_feature, (1, 1, bs * num_anchor, self.embed_dims)
        )
        learnable = ttnn.linear(
            inst_flat, self.learnable_fc_weight, bias=self.learnable_fc_bias,
            compute_kernel_config=self._hifi_compute_config,
        )  # [1, 1, bs*num_anchor, 18]
        logger.debug("_kps_generator: learnable linear done")

        learnable = ttnn.reshape(
            learnable, (bs * num_anchor, self.num_learnable_pts, 3)
        )  # [bs*num_anchor, 6, 3]
        learnable = ttnn.sigmoid(learnable)
        learnable = ttnn.subtract(learnable, self._scalar_half)  # sigmoid - 0.5
        learnable_kps = ttnn.multiply(learnable, size_3d)  # [bs*num_anchor, 6, 3]
        ttnn.deallocate(learnable)
        # Note: size_3d is a reshape (view) of size, don't deallocate separately
        logger.debug("_kps_generator: learnable_kps done")

        # Concat fixed + learnable: [bs*num_anchor, 13, 3]
        key_points = ttnn.concat([fix_kps, learnable_kps], dim=1)
        ttnn.deallocate(fix_kps)
        ttnn.deallocate(learnable_kps)
        logger.debug("_kps_generator: concat done")

        # --- Rotation + translation on device ---
        n = bs * num_anchor
        anchor_flat = ttnn.reshape(anchor, (n, 1, 11))

        cos_yaw = ttnn.slice(anchor_flat, [0, 0, COS_YAW], [n, 1, COS_YAW + 1])  # [n, 1, 1]
        sin_yaw = ttnn.slice(anchor_flat, [0, 0, SIN_YAW], [n, 1, SIN_YAW + 1])  # [n, 1, 1]

        kp_x = ttnn.slice(key_points, [0, 0, 0], [n, self.num_pts, 1])  # [n, 13, 1]
        kp_y = ttnn.slice(key_points, [0, 0, 1], [n, self.num_pts, 2])  # [n, 13, 1]
        kp_z = ttnn.slice(key_points, [0, 0, 2], [n, self.num_pts, 3])  # [n, 13, 1]
        ttnn.deallocate(key_points)

        rot_x = ttnn.subtract(ttnn.multiply(cos_yaw, kp_x), ttnn.multiply(sin_yaw, kp_y))
        rot_y = ttnn.add(ttnn.multiply(sin_yaw, kp_x), ttnn.multiply(cos_yaw, kp_y))
        ttnn.deallocate(kp_x); ttnn.deallocate(kp_y)

        key_points = ttnn.concat([rot_x, rot_y, kp_z], dim=-1)  # [n, 13, 3]
        ttnn.deallocate(rot_x); ttnn.deallocate(rot_y); ttnn.deallocate(kp_z)
        ttnn.deallocate(anchor_flat)

        center = ttnn.reshape(anchor, (n, 1, 11))
        center = ttnn.slice(center, [0, 0, X], [n, 1, Z + 1])  # [n, 1, 3]
        key_points = ttnn.add(key_points, center)
        ttnn.deallocate(center)

        key_points = ttnn.reshape(key_points, (bs, n * self.num_pts // bs, 3))
        logger.debug("_kps_generator: rotation+translate (device) done")

        return key_points

    def _project_points(
        self,
        key_points: ttnn.Tensor,
        projection_mat: ttnn.Tensor,
        image_wh: ttnn.Tensor,
        bs: int,
        num_anchor: int,
    ) -> ttnn.Tensor:
        """Project 3D key points to normalized 2D per camera on device.

        Uses batched matmul across all cameras (no per-camera loop).

        Args:
            key_points: [bs, num_anchor*num_pts, 3] on device
            projection_mat: [bs, num_cams, 4, 4] on device
            image_wh: [bs, num_cams, 2] on device

        Returns:
            points_2d_grid: [bs*num_cams, num_anchor, num_pts, 2] on device
        """
        n_pts_total = num_anchor * self.num_pts
        nc = self.num_cams

        # Append ones for homogeneous: [bs, n_pts_total, 4]
        _kw_f32 = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32)
        if self._mesh_device is not None:
            _kw_f32["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        if key_points.dtype != ttnn.float32:
            key_points = ttnn.typecast(key_points, ttnn.float32)
        ones = ttnn.from_torch(torch.ones(bs, n_pts_total, 1), **_kw_f32)
        pts_homo = ttnn.concat([key_points, ones], dim=-1)  # [bs, n_pts_total, 4]
        logger.debug(f"DFA _project_points: pts_homo shape={pts_homo.shape}")

        # Batched projection: all cameras in one matmul
        # Expand pts_homo: [bs, n_pts, 4] → [bs*nc, n_pts, 4]
        pts_expanded = ttnn.concat([pts_homo] * nc, dim=0)  # [bs*nc, n_pts, 4]
        ttnn.deallocate(pts_homo)
        ttnn.deallocate(ones)

        # Reshape proj: [bs, nc, 4, 4] → [bs*nc, 4, 4] → transpose → [bs*nc, 4, 4]
        proj = ttnn.reshape(projection_mat, (bs * nc, 4, 4))
        proj_t = ttnn.transpose(proj, -2, -1)

        # Batched matmul: [bs*nc, n_pts, 4] × [bs*nc, 4, 4] → [bs*nc, n_pts, 4]
        projected = ttnn.matmul(pts_expanded, proj_t, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(pts_expanded)
        logger.debug("DFA _project_points: batched matmul done")

        # Perspective divide (all cameras at once)
        xy = ttnn.slice(projected, [0, 0, 0], [bs * nc, n_pts_total, 2])
        z = ttnn.slice(projected, [0, 0, 2], [bs * nc, n_pts_total, 3])
        ttnn.deallocate(projected)
        z_clamped = ttnn.clamp(z, min=1e-5)
        z_recip = ttnn.reciprocal(z_clamped)
        xy_div = ttnn.multiply(xy, z_recip)
        ttnn.deallocate(xy); ttnn.deallocate(z); ttnn.deallocate(z_clamped); ttnn.deallocate(z_recip)
        logger.debug("DFA _project_points: perspective divide done")

        # Grid normalization (all cameras at once)
        # image_wh: [bs, nc, 2] → [bs*nc, 1, 2]
        wh = ttnn.reshape(image_wh, (bs * nc, 1, 2))
        wh_recip = ttnn.reciprocal(wh)
        xy_norm = ttnn.multiply(xy_div, wh_recip)
        ttnn.deallocate(xy_div); ttnn.deallocate(wh_recip)

        xy_scaled = ttnn.multiply(xy_norm, self._scalar_two)
        xy_grid = ttnn.subtract(xy_scaled, self._scalar_one)
        ttnn.deallocate(xy_norm); ttnn.deallocate(xy_scaled)
        logger.debug("DFA _project_points: grid normalization done")

        # Reshape to [bs*nc, num_anchor, num_pts, 2]
        points_2d = ttnn.reshape(xy_grid, (bs * nc, num_anchor, self.num_pts, 2))

        return points_2d

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

        # Linear1: [bs*num_cams, 12] -> [bs*num_cams, 256]
        x = ttnn.linear(cam_input, self.cam_linear1_weight, bias=self.cam_linear1_bias, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(cam_input)
        relu_in = x
        x = ttnn.relu(x)
        relu_out = x
        x = ttnn.layer_norm(x, weight=self.cam_ln1_weight, bias=self.cam_ln1_bias, epsilon=1e-5, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(relu_in)
        ttnn.deallocate(relu_out)

        # Linear2: [bs*num_cams, 256] -> [bs*num_cams, 256]
        linear2_in = x
        x = ttnn.linear(x, self.cam_linear2_weight, bias=self.cam_linear2_bias, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(linear2_in)
        relu_in = x
        x = ttnn.relu(x)
        relu_out = x
        x = ttnn.layer_norm(x, weight=self.cam_ln2_weight, bias=self.cam_ln2_bias, epsilon=1e-5, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(relu_in)
        ttnn.deallocate(relu_out)

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
            camera_embed = self._camera_encoder(projection_mat, bs)
            feat_exp = ttnn.reshape(feature, (bs, num_anchor, 1, self.embed_dims))
            cam_exp = ttnn.reshape(
                camera_embed, (bs, 1, self.num_cams, self.embed_dims)
            )
            feature = ttnn.add(feat_exp, cam_exp)
            ttnn.deallocate(camera_embed)
            feature = ttnn.reshape(
                feature, (1, 1, bs * num_anchor * self.num_cams, self.embed_dims)
            )
        else:
            feature = ttnn.reshape(feature, (1, 1, bs * num_anchor, self.embed_dims))

        weights = ttnn.linear(
            feature, self.weights_fc_weight, bias=self.weights_fc_bias,
            compute_kernel_config=self._hifi_compute_config,
        )

        ttnn.deallocate(feature)

        total_clp = self.num_cams * self.num_levels * self.num_pts
        if self.use_camera_embed:
            weights = ttnn.reshape(
                weights,
                (1, 1, bs * num_anchor,
                 self.num_cams * self.num_levels * self.num_pts * self.num_groups),
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

        weights = ttnn.softmax(weights, dim=1, numeric_stable=True,
                               compute_kernel_config=self._hifi_compute_config)

        return weights

    def _feature_sampling(
        self,
        feature_maps: List[ttnn.Tensor],
        points_2d: ttnn.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        bs: int,
        num_anchor: int,
    ) -> ttnn.Tensor:
        """Sample features from FPN maps.

        Weight order from _get_weights: for each cam, for each level, for each pt.
        So we need: cam0_lvl0_pts, cam0_lvl1_pts, ..., cam0_lvl3_pts, cam1_lvl0_pts, ...

        Args:
            feature_maps: List of ttnn.Tensor [1, 1, N*H*W, C] from FPN (on device)
            points_2d: [bs*num_cams, num_anchor, num_pts, 2] on device
            spatial_shapes: [(H, W)] per FPN level

        Returns:
            features: [bs*num_anchor, num_cams*num_levels*num_pts, embed_dims] on device
        """
        n_batch = bs * self.num_cams

        # Pre-convert grid once (reused across all levels)
        grid = ttnn.to_layout(points_2d, ttnn.ROW_MAJOR_LAYOUT)
        grid = ttnn.to_memory_config(grid, ttnn.DRAM_MEMORY_CONFIG)
        if grid.dtype != ttnn.float32:
            grid = ttnn.typecast(grid, ttnn.float32)

        # 1. grid_sample_lerp per level
        all_level_features = []
        for level_idx, fm_tt in enumerate(feature_maps):
            h, w = spatial_shapes[level_idx]

            # Reshape feature map: [1, 1, N*H*W, C] -> [N, H, W, C] (NHWC)
            fm = ttnn.to_memory_config(fm_tt, ttnn.DRAM_MEMORY_CONFIG)
            fm = ttnn.to_layout(fm, ttnn.ROW_MAJOR_LAYOUT)
            fm = ttnn.reshape(fm, (n_batch, h, w, self.embed_dims))

            sampled = ttnn.grid_sample_lerp(fm, grid, padding_mode="zeros", align_corners=False)
            sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
            sampled = ttnn.to_memory_config(sampled, ttnn.DRAM_MEMORY_CONFIG)
            all_level_features.append(sampled)

        ttnn.deallocate(grid)

        # 2. Rearrange via slice + concat (no host transfer)
        #
        # Each level result: [bs*num_cams, num_anchor, num_pts, embed_dims]
        #   dim0 layout: [cam0_bs, cam1_bs, ..., cam5_bs]  (cam varies fastest? no, bs*cams)
        #   actually: dim0 = bs*num_cams, ordered as [b0c0, b0c1, ..., b0c5, b1c0, ...]
        #
        # We need final: [bs*num_anchor, cams*levels*pts, embed_dims]
        #   with order: cam0_lvl0_pts, cam0_lvl1_pts, ..., cam0_lvl3_pts, cam1_lvl0_pts, ...
        #
        # Strategy:
        #   For each cam: slice cam's data from each level → concat levels
        #   Then concat all cams
        #   Finally reshape to merge anchor into batch dim

        chunks = []
        for cam_idx in range(self.num_cams):
            for level_idx in range(self.num_levels):
                sampled = all_level_features[level_idx]
                # [bs*num_cams, num_anchor, num_pts, embed_dims]
                # cam_idx's slice: rows [cam_idx*bs : (cam_idx+1)*bs]
                # But dim0 order is [b0c0, b0c1, ..., b0c5, b1c0, ...]
                # i.e. for bs=1: [c0, c1, c2, c3, c4, c5]
                # for bs=2: [b0c0, b0c1, ..., b0c5, b1c0, ..., b1c5]
                # So cam_idx for batch b is at index b*num_cams + cam_idx

                # For general bs, we need to gather all batches for this cam.
                # With bs=1 (typical inference), cam_idx maps to row cam_idx directly.
                # For bs>1, we slice each batch separately and concat.
                if bs == 1:
                    chunk = ttnn.slice(
                        sampled,
                        [cam_idx, 0, 0, 0],
                        [cam_idx + 1, num_anchor, self.num_pts, self.embed_dims],
                    )
                    # [1, num_anchor, num_pts, embed_dims]
                    chunks.append(chunk)
                else:
                    for b in range(bs):
                        row = b * self.num_cams + cam_idx
                        chunk = ttnn.slice(
                            sampled,
                            [row, 0, 0, 0],
                            [row + 1, num_anchor, self.num_pts, self.embed_dims],
                        )
                        # [1, num_anchor, num_pts, embed_dims]
                        chunks.append(chunk)

        # Concat all chunks along pts dim (dim=2)
        # Each chunk: [1, num_anchor, num_pts, embed_dims]  (bs=1)
        # or [1, num_anchor, num_pts, embed_dims] per batch (bs>1)
        #
        # For bs=1: 6 cams * 4 levels = 24 chunks
        #   concat dim=2 → [1, num_anchor, 24*num_pts, embed_dims]
        #   = [1, num_anchor, cams*levels*pts, embed_dims]
        if bs == 1:
            features = ttnn.concat(chunks, dim=2)
            for c in chunks:
                ttnn.deallocate(c)
            # [1, num_anchor, num_cams*num_levels*num_pts, embed_dims]
            features = ttnn.reshape(
                features,
                (
                    num_anchor,
                    self.num_cams * self.num_levels * self.num_pts,
                    self.embed_dims,
                ),
            )
        else:
            # Group chunks by batch: each batch has cams*levels chunks
            batch_features = []
            for b in range(bs):
                # chunks for batch b: indices [b, b+bs, b+2*bs, ...]
                # Actually chunks are ordered: cam0_lvl0_b0, cam0_lvl0_b1, ..., cam0_lvl1_b0, ...
                # Let's re-index: for cam c, level l, batch b → index = (c*num_levels + l)*bs + b
                b_chunks = []
                for c in range(self.num_cams):
                    for l in range(self.num_levels):
                        idx = (c * self.num_levels + l) * bs + b
                        b_chunks.append(chunks[idx])
                b_feat = ttnn.concat(b_chunks, dim=2)
                # [1, num_anchor, cams*levels*pts, embed_dims]
                batch_features.append(b_feat)
            features = ttnn.concat(batch_features, dim=0)
            # [bs, num_anchor, cams*levels*pts, embed_dims]
            features = ttnn.reshape(
                features,
                (
                    bs * num_anchor,
                    self.num_cams * self.num_levels * self.num_pts,
                    self.embed_dims,
                ),
            )

        # Note: do NOT deallocate all_level_features here — sliced chunks
        # may still share underlying memory with the concat result.
        # for sampled in all_level_features:
        #     ttnn.deallocate(sampled)

        return features

    def _multi_view_level_fusion(
        self,
        features: ttnn.Tensor,
        weights: ttnn.Tensor,
        bs: int,
        num_anchor: int,
    ) -> ttnn.Tensor:
        """Weighted fusion on device via repeat_interleave + element-wise multiply + sum.

        Args:
            features: [bs*num_anchor, cams*levels*pts, embed_dims] on device
            weights: [bs*num_anchor, cams*levels*pts, num_groups] on device

        Returns:
            output: [1, 1, bs*num_anchor, embed_dims] on device
        """
        total_clp = self.num_cams * self.num_levels * self.num_pts
        n = bs * num_anchor  # 900

        # Expand weights [n, clp, G] → [n, clp, embed_dims] by repeating each group value D times
        # repeat_interleave on 2D: [n*clp, G] → [n*clp, G*D=embed_dims]
        weights = ttnn.reshape(weights, (n * total_clp, self.num_groups))
        weights = ttnn.repeat_interleave(weights, self.group_dims, dim=-1)
        weights = ttnn.reshape(weights, (n, total_clp, self.embed_dims))

        # Element-wise multiply + sum over clp dimension
        features = ttnn.multiply(features, weights)
        ttnn.deallocate(weights)
        features = ttnn.sum(features, dim=1)

        # Reshape: [n, 1, embed_dims] → [1, 1, n, embed_dims]
        features = ttnn.reshape(features, (1, 1, n, self.embed_dims))

        return features

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
        """Full forward pass of DeformableFeatureAggregation on device.

        Args:
            instance_feature: [bs, num_anchor, embed_dims] on device (TILE)
            anchor: [bs, num_anchor, 11] on device (TILE)
            anchor_embed: [bs, num_anchor, embed_dims] on device (TILE)
            feature_maps: List of ttnn.Tensor from FPN [1, 1, N*H*W, C]
            projection_mat: [bs, num_cams, 4, 4] on device (TILE)
            image_wh: [bs, num_cams, 2] on device (TILE)
            spatial_shapes: [(H, W)] per FPN level
            bs: batch size
            num_anchor: number of anchors

        Returns:
            output: [bs, num_anchor, embed_dims] (add) or
                    [bs, num_anchor, 2*embed_dims] (cat) on device
        """
        # 1. Generate 3D key points
        logger.debug("DFA run: _kps_generator start")
        key_points = self._kps_generator(anchor, instance_feature, bs, num_anchor)
        logger.debug("DFA run: _kps_generator done")
        # [bs, num_anchor*num_pts, 3]

        # 2. Project to 2D per camera
        logger.debug("DFA run: _project_points start")
        points_2d = self._project_points(
            key_points, projection_mat, image_wh, bs, num_anchor
        )
        ttnn.deallocate(key_points)
        # [bs*num_cams, num_anchor, num_pts, 2]

        # 3. Compute attention weights
        logger.debug("DFA run: _get_weights start")
        weights = self._get_weights(
            instance_feature, anchor_embed, projection_mat, bs, num_anchor
        )
        logger.debug("DFA run: _get_weights done")
        # [bs*num_anchor, num_cams*num_levels*num_pts, num_groups]

        # 4+5. Feature sampling (grid_sample_lerp) + fusion
        logger.debug("DFA run: _feature_sampling start")
        features = self._feature_sampling(
            feature_maps, points_2d, spatial_shapes, bs, num_anchor
        )
        logger.debug("DFA run: _feature_sampling done")
        logger.debug("DFA run: _multi_view_level_fusion start")
        features = self._multi_view_level_fusion(features, weights, bs, num_anchor)
        logger.debug("DFA run: _multi_view_level_fusion done")
        ttnn.deallocate(points_2d)

        # In mesh SPMD mode: each device has partial fusion (3 cams).
        # Combine via host (all_reduce hangs on N300).
        if self._mesh_device is not None:
            logger.debug("DFA run: mesh combine start")
            composer = ttnn.ConcatMeshToTensor(self._mesh_device, dim=0)
            partials = ttnn.to_torch(features, mesh_composer=composer).float()
            combined = partials[:1] + partials[1:]
            del partials
            ttnn.deallocate(features)
            features = ttnn.from_torch(
                combined, layout=ttnn.TILE_LAYOUT, device=self._mesh_device,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self._mesh_device),
            )
            del combined
            logger.debug("DFA run: mesh combine done")

        logger.debug("DFA run: output_proj start")
        output = ttnn.linear(
            features, self.output_proj_weight, bias=self.output_proj_bias,
            compute_kernel_config=self._hifi_compute_config,
        )
        ttnn.deallocate(features)
        logger.debug("DFA run: output_proj done")

        logger.debug("DFA run: residual start")
        inst_flat = ttnn.reshape(
            instance_feature, (1, 1, bs * num_anchor, self.embed_dims)
        )
        if output.dtype != inst_flat.dtype:
            output = ttnn.typecast(output, inst_flat.dtype)
        if self.residual_mode == "add":
            output = ttnn.add(output, inst_flat)
            output = ttnn.reshape(output, (bs, num_anchor, self.embed_dims))
        elif self.residual_mode == "cat":
            output = ttnn.concat([output, inst_flat], dim=-1)
            output = ttnn.reshape(output, (bs, num_anchor, 2 * self.embed_dims))
        logger.debug("DFA run: residual done")

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
