# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# DeformableFeatureAggregation for TT Devices (Device-Only)
#
# All operations run on TT device using ttnn ops. No host-device transfers
# during forward pass except final output retrieval.
#
# Forward flow:
#   1. kps_generator: anchor → 3D key points (ttnn ops)
#   2. project_points: 3D → 2D via projection matrix (ttnn.matmul)
#   3. get_weights: instance_feature → attention weights (ttnn.linear + softmax)
#   4. feature_sampling: grid_sample per FPN level (ttnn.grid_sample)
#   5. multi_view_level_fusion: weighted sum (ttnn.multiply + ttnn.sum)
#   6. output_proj: ttnn.linear + residual
# =============================================================================

from typing import Dict, List, Tuple
import time

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
    ) -> None:
        self.device = device
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
        if tensor.dim() == 2:
            # Pad to tile-aligned if needed
            pass
        t = ttnn.from_torch(tensor.float(), layout=ttnn.TILE_LAYOUT, device=self.device)
        return t

    def _to_device_bias(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Move bias tensor to device as [1, 1, 1, N] in TILE layout."""
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        t = ttnn.from_torch(tensor.float(), layout=ttnn.TILE_LAYOUT, device=self.device)
        return t

    def _to_device_1d(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Move 1D tensor (LayerNorm weight/bias) to device as [1, 1, 1, N]."""
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        t = ttnn.from_torch(tensor.float(), layout=ttnn.TILE_LAYOUT, device=self.device)
        return t

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
        size_wlh = ttnn.slice(
            anchor, [0, 0, W], [bs, num_anchor, H + 1]
        )  # [bs, num_anchor, 3]
        size = ttnn.exp(size_wlh)  # [bs, num_anchor, 3]

        # Reshape size for broadcasting: [bs*num_anchor, 1, 3]
        size_3d = ttnn.reshape(size, (bs * num_anchor, 1, 3))

        # Fixed key points: fix_scale [7, 3] * size [bs*num_anchor, 1, 3]
        # Broadcast multiply
        fix_scale_dev = self.fix_scale  # [7, 3] on device
        # Expand fix_scale to [1, 7, 3] for broadcast with [bs*num_anchor, 1, 3]
        fix_scale_3d = ttnn.reshape(fix_scale_dev, (1, 7, 3))

        # size_for_fix: [bs*num_anchor, 1, 3] broadcast with [1, 7, 3]
        fix_kps = ttnn.multiply(fix_scale_3d, size_3d)  # [bs*num_anchor, 7, 3]

        # Learnable key points: linear(instance_feature) -> sigmoid - 0.5
        # instance_feature: [bs, num_anchor, 256] -> flatten to [bs*num_anchor, 1, 256]
        inst_flat = ttnn.reshape(
            instance_feature, (1, 1, bs * num_anchor, self.embed_dims)
        )
        learnable = ttnn.linear(
            inst_flat, self.learnable_fc_weight, bias=self.learnable_fc_bias
        )  # [1, 1, bs*num_anchor, 18]
        learnable = ttnn.reshape(
            learnable, (bs * num_anchor, self.num_learnable_pts, 3)
        )  # [bs*num_anchor, 6, 3]
        learnable = ttnn.sigmoid(learnable)
        half = ttnn.from_torch(
            torch.full((1, 1, 1), 0.5),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        learnable = ttnn.subtract(learnable, half)  # sigmoid - 0.5

        # learnable * size: [bs*num_anchor, 6, 3] * [bs*num_anchor, 1, 3]
        learnable_kps = ttnn.multiply(learnable, size_3d)  # [bs*num_anchor, 6, 3]

        # Concat fixed + learnable: [bs*num_anchor, 13, 3]
        key_points = ttnn.concat([fix_kps, learnable_kps], dim=1)

        # Rotate by yaw angle (element-wise, no rotation matrix needed)
        # cos_yaw, sin_yaw from anchor[:, :, 6:8]
        cos_yaw = ttnn.slice(anchor, [0, 0, COS_YAW], [bs, num_anchor, COS_YAW + 1])
        sin_yaw = ttnn.slice(anchor, [0, 0, SIN_YAW], [bs, num_anchor, SIN_YAW + 1])
        # [bs, num_anchor, 1] -> [bs*num_anchor, 1, 1] for broadcast
        cos_yaw = ttnn.reshape(cos_yaw, (bs * num_anchor, 1, 1))
        sin_yaw = ttnn.reshape(sin_yaw, (bs * num_anchor, 1, 1))

        # kp_x = key_points[..., 0:1], kp_y = key_points[..., 1:2], kp_z = key_points[..., 2:3]
        kp_x = ttnn.slice(key_points, [0, 0, 0], [bs * num_anchor, self.num_pts, 1])
        kp_y = ttnn.slice(key_points, [0, 0, 1], [bs * num_anchor, self.num_pts, 2])
        kp_z = ttnn.slice(key_points, [0, 0, 2], [bs * num_anchor, self.num_pts, 3])

        # rot_x = cos*kp_x - sin*kp_y
        # rot_y = sin*kp_x + cos*kp_y
        rot_x = ttnn.subtract(
            ttnn.multiply(cos_yaw, kp_x), ttnn.multiply(sin_yaw, kp_y)
        )
        rot_y = ttnn.add(ttnn.multiply(sin_yaw, kp_x), ttnn.multiply(cos_yaw, kp_y))
        # rot_z = kp_z (unchanged)

        # Concat rotated: [bs*num_anchor, num_pts, 3]
        key_points = ttnn.concat([rot_x, rot_y, kp_z], dim=-1)

        # Translate to anchor center: anchor[..., 0:3] = [X, Y, Z]
        center = ttnn.slice(anchor, [0, 0, X], [bs, num_anchor, Z + 1])
        center = ttnn.reshape(center, (bs * num_anchor, 1, 3))
        key_points = ttnn.add(key_points, center)

        # Reshape to [bs, num_anchor, num_pts, 3]
        # For matmul in projection we keep [bs, num_anchor*num_pts, 3]
        key_points = ttnn.reshape(key_points, (bs, num_anchor * self.num_pts, 3))

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

        Args:
            key_points: [bs, num_anchor*num_pts, 3] on device
            projection_mat: [bs, num_cams, 4, 4] on device
            image_wh: [bs, num_cams, 2] on device

        Returns:
            points_2d_grid: [bs*num_cams, num_anchor, num_pts, 2] on device (ROW_MAJOR, float32)
        """
        n_pts_total = num_anchor * self.num_pts

        # Append ones for homogeneous: [bs, n_pts_total, 4]
        ones = ttnn.from_torch(
            torch.ones(bs, n_pts_total, 1),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        pts_homo = ttnn.concat([key_points, ones], dim=-1)  # [bs, n_pts_total, 4]

        # Per-camera projection via loop (avoid 6D tensor)
        all_cam_points = []
        for cam_idx in range(self.num_cams):
            # proj: [bs, 4, 4] for this camera
            proj = ttnn.slice(
                projection_mat,
                [0, cam_idx, 0, 0],
                [bs, cam_idx + 1, 4, 4],
            )  # [bs, 1, 4, 4]
            proj = ttnn.reshape(proj, (bs, 4, 4))

            # pts_homo: [bs, n_pts_total, 4]
            # result = pts_homo @ proj^T -> [bs, n_pts_total, 4]
            proj_t = ttnn.transpose(proj, -2, -1)  # [bs, 4, 4]
            projected = ttnn.matmul(pts_homo, proj_t)  # [bs, n_pts_total, 4]

            # Perspective divide: xy / max(z, 1e-5)
            xy = ttnn.slice(projected, [0, 0, 0], [bs, n_pts_total, 2])
            z = ttnn.slice(projected, [0, 0, 2], [bs, n_pts_total, 3])
            z_clamped = ttnn.clamp(z, min=1e-5)
            xy_div = ttnn.multiply(
                xy, ttnn.reciprocal(z_clamped)
            )  # [bs, n_pts_total, 2]

            # Normalize by image_wh: [bs, 1, 2] for this cam
            wh = ttnn.slice(image_wh, [0, cam_idx, 0], [bs, cam_idx + 1, 2])
            wh = ttnn.reshape(wh, (bs, 1, 2))
            xy_norm = ttnn.multiply(
                xy_div, ttnn.reciprocal(wh)
            )  # [bs, n_pts_total, 2] in [0,1]

            # Convert to grid_sample range [-1, 1]
            two = ttnn.from_torch(
                torch.full((1, 1, 1), 2.0),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            one = ttnn.from_torch(
                torch.full((1, 1, 1), 1.0),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            xy_grid = ttnn.subtract(ttnn.multiply(xy_norm, two), one)
            # [bs, n_pts_total, 2]

            # Reshape to [bs, num_anchor, num_pts, 2]
            xy_grid = ttnn.reshape(xy_grid, (bs, num_anchor, self.num_pts, 2))
            all_cam_points.append(xy_grid)

        # Stack cameras: concat along dim0 after reshape
        # Each is [bs, num_anchor, num_pts, 2]
        # We want [bs*num_cams, num_anchor, num_pts, 2]
        points_2d = ttnn.concat(all_cam_points, dim=0)
        # [bs*num_cams, num_anchor, num_pts, 2]

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
        x = ttnn.linear(cam_input, self.cam_linear1_weight, bias=self.cam_linear1_bias)
        x = ttnn.relu(x)
        x = ttnn.layer_norm(x, weight=self.cam_ln1_weight, bias=self.cam_ln1_bias)

        # Linear2: [bs*num_cams, 256] -> [bs*num_cams, 256]
        x = ttnn.linear(x, self.cam_linear2_weight, bias=self.cam_linear2_bias)
        x = ttnn.relu(x)
        x = ttnn.layer_norm(x, weight=self.cam_ln2_weight, bias=self.cam_ln2_bias)

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
            # [bs, num_cams, 256]

            # feature[:, :, None] + camera_embed[:, None]
            # -> [bs, num_anchor, num_cams, 256]
            # Reshape for broadcast add:
            feat_exp = ttnn.reshape(feature, (bs, num_anchor, 1, self.embed_dims))
            cam_exp = ttnn.reshape(
                camera_embed, (bs, 1, self.num_cams, self.embed_dims)
            )
            feature = ttnn.add(feat_exp, cam_exp)
            # [bs, num_anchor, num_cams, 256]

            # Flatten for linear: [1, 1, bs*num_anchor*num_cams, 256]
            feature = ttnn.reshape(
                feature, (1, 1, bs * num_anchor * self.num_cams, self.embed_dims)
            )
        else:
            feature = ttnn.reshape(feature, (1, 1, bs * num_anchor, self.embed_dims))

        # weights_fc: -> [..., 416] where 416 = num_groups * num_levels * num_pts
        weights = ttnn.linear(
            feature, self.weights_fc_weight, bias=self.weights_fc_bias
        )

        # Reshape for softmax: [..., num_cams*num_levels*num_pts, num_groups]
        total_clp = self.num_cams * self.num_levels * self.num_pts
        if self.use_camera_embed:
            # [1, 1, bs*num_anchor*num_cams, 416]
            # 416 = num_levels * num_pts * num_groups = 4 * 13 * 8
            # Reshape to [bs*num_anchor, num_cams*num_levels*num_pts, num_groups]
            weights = ttnn.reshape(
                weights,
                (
                    1,
                    1,
                    bs * num_anchor,
                    self.num_cams * self.num_levels * self.num_pts * self.num_groups,
                ),
            )
            # For softmax over cams*levels*pts dim:
            # [bs*num_anchor, cams*levels*pts, groups]
            weights = ttnn.reshape(
                weights, (bs * num_anchor, total_clp, self.num_groups)
            )
        else:
            weights = ttnn.reshape(
                weights, (bs * num_anchor, total_clp, self.num_groups)
            )

        # Softmax over dim=1 (cams*levels*pts)
        weights = ttnn.softmax(weights, dim=1)

        return weights

    def _feature_sampling(
        self,
        feature_maps: List[ttnn.Tensor],
        points_2d: ttnn.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        bs: int,
        num_anchor: int,
    ) -> ttnn.Tensor:
        """Sample features from FPN maps on device (fully device-only).

        Uses per-camera slice + concat to rearrange without host transfer.
        grid_sample results [bs*num_cams, num_anchor, num_pts, embed_dims] are
        sliced per camera, then interleaved per level via concat.

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

        # 1. grid_sample per level
        all_level_features = []
        for level_idx, fm_tt in enumerate(feature_maps):
            h, w = spatial_shapes[level_idx]
            logger.debug(f"==== DFA grid_sample level {level_idx}: spatial={h}x{w}")

            # Reshape feature map: [1, 1, N*H*W, C] -> [N, H, W, C] (NHWC)
            fm = ttnn.to_memory_config(fm_tt, ttnn.DRAM_MEMORY_CONFIG)
            fm = ttnn.to_layout(fm, ttnn.ROW_MAJOR_LAYOUT)
            fm = ttnn.reshape(fm, (n_batch, h, w, self.embed_dims))

            # Grid: [bs*num_cams, num_anchor, num_pts, 2]
            grid = ttnn.to_layout(points_2d, ttnn.ROW_MAJOR_LAYOUT)
            grid = ttnn.to_memory_config(grid, ttnn.DRAM_MEMORY_CONFIG)

            # grid_sample (NHWC)
            sampled = ttnn.grid_sample(
                fm,
                grid,
                mode="bilinear",
                align_corners=False,
                padding_mode="zeros",
            )
            # [bs*num_cams, num_anchor, num_pts, embed_dims]

            sampled = ttnn.to_layout(sampled, ttnn.TILE_LAYOUT)
            sampled = ttnn.to_memory_config(sampled, ttnn.DRAM_MEMORY_CONFIG)

            all_level_features.append(sampled)

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

        # Deallocate grid_sample results
        for sampled in all_level_features:
            ttnn.deallocate(sampled)

        return features

    def _multi_view_level_fusion(
        self,
        features: ttnn.Tensor,
        weights: ttnn.Tensor,
        bs: int,
        num_anchor: int,
    ) -> ttnn.Tensor:
        """Weighted fusion on device.

        Args:
            features: [bs*num_anchor, cams*levels*pts, embed_dims] on device
            weights: [bs*num_anchor, cams*levels*pts, num_groups] on device

        Returns:
            output: [bs*num_anchor, embed_dims] on device
        """
        total_clp = self.num_cams * self.num_levels * self.num_pts

        # Split embed_dims into groups: [bs*num_anchor, clp, num_groups, group_dims]
        features = ttnn.reshape(
            features, (bs * num_anchor, total_clp, self.num_groups, self.group_dims)
        )

        # Expand weights: [bs*num_anchor, clp, num_groups, 1]
        weights = ttnn.reshape(
            weights, (bs * num_anchor, total_clp, self.num_groups, 1)
        )

        # Weighted features
        features = ttnn.multiply(features, weights)
        # [bs*num_anchor, clp, num_groups, group_dims]

        # Sum over clp dimension (dim=1)
        features = ttnn.sum(features, dim=1)
        # [bs*num_anchor, 1, num_groups, group_dims]

        # Reshape: [bs*num_anchor, num_groups*group_dims] = [bs*num_anchor, embed_dims]
        features = ttnn.reshape(features, (1, 1, bs * num_anchor, self.embed_dims))

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
        key_points = self._kps_generator(anchor, instance_feature, bs, num_anchor)
        # [bs, num_anchor*num_pts, 3]

        # 2. Project to 2D per camera
        points_2d = self._project_points(
            key_points, projection_mat, image_wh, bs, num_anchor
        )
        # [bs*num_cams, num_anchor, num_pts, 2]

        # 3. Compute attention weights
        weights = self._get_weights(
            instance_feature, anchor_embed, projection_mat, bs, num_anchor
        )
        # [bs*num_anchor, num_cams*num_levels*num_pts, num_groups]

        # 4. Feature sampling (grid_sample on device)
        features = self._feature_sampling(
            feature_maps, points_2d, spatial_shapes, bs, num_anchor
        )
        # [bs*num_anchor, num_cams*num_levels*num_pts, embed_dims]

        # 5. Multi-view level fusion
        features = self._multi_view_level_fusion(features, weights, bs, num_anchor)
        # [1, 1, bs*num_anchor, embed_dims]

        # 6. Output projection
        output = ttnn.linear(
            features, self.output_proj_weight, bias=self.output_proj_bias
        )
        # [1, 1, bs*num_anchor, embed_dims]

        # 7. Residual
        inst_flat = ttnn.reshape(
            instance_feature, (1, 1, bs * num_anchor, self.embed_dims)
        )
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
