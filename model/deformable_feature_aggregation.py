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

        # HiFi compute config for precision-sensitive operations
        self._hifi_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        # Pre-allocate scalar constants on device (reused per camera in _project_points)
        self._scalar_two = ttnn.from_torch(
            torch.full((1, 1, 1), 2.0),
            layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32,
        )
        self._scalar_one = ttnn.from_torch(
            torch.full((1, 1, 1), 1.0),
            layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32,
        )
        self._scalar_half = ttnn.from_torch(
            torch.full((1, 1, 1), 0.5),
            layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32,
        )

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
        t = ttnn.from_torch(tensor.float(), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32)
        return t

    def _to_device_bias(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Move bias tensor to device as [1, 1, 1, N] in TILE layout."""
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        t = ttnn.from_torch(tensor.float(), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32)
        return t

    def _to_device_1d(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Move 1D tensor (LayerNorm weight/bias) to device as [1, 1, 1, N]."""
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        t = ttnn.from_torch(tensor.float(), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.float32)
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
        logger.debug(f"_kps_generator: slice+exp, anchor shape={anchor.shape}")
        size_wlh = ttnn.slice(
            anchor, [0, 0, W], [bs, num_anchor, H + 1]
        )  # [bs, num_anchor, 3]
        size = ttnn.exp(size_wlh)  # [bs, num_anchor, 3]
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(size_wlh)
        logger.debug("_kps_generator: slice+exp done")

        # Reshape size for broadcasting: [bs*num_anchor, 1, 3]
        size_3d = ttnn.reshape(size, (bs * num_anchor, 1, 3))

        # Fixed key points: fix_scale [7, 3] * size [bs*num_anchor, 1, 3]
        fix_scale_3d = ttnn.reshape(self.fix_scale, (1, 7, 3))
        fix_kps = ttnn.multiply(fix_scale_3d, size_3d)  # [bs*num_anchor, 7, 3]
        ttnn.synchronize_device(self.device)
        logger.debug("_kps_generator: fix_kps done")

        # Learnable key points
        inst_flat = ttnn.reshape(
            instance_feature, (1, 1, bs * num_anchor, self.embed_dims)
        )
        learnable = ttnn.linear(
            inst_flat, self.learnable_fc_weight, bias=self.learnable_fc_bias,
            compute_kernel_config=self._hifi_compute_config,
        )  # [1, 1, bs*num_anchor, 18]
        ttnn.synchronize_device(self.device)
        logger.debug("_kps_generator: learnable linear done")

        learnable = ttnn.reshape(
            learnable, (bs * num_anchor, self.num_learnable_pts, 3)
        )  # [bs*num_anchor, 6, 3]
        learnable = ttnn.sigmoid(learnable)
        learnable = ttnn.subtract(learnable, self._scalar_half)  # sigmoid - 0.5
        learnable_kps = ttnn.multiply(learnable, size_3d)  # [bs*num_anchor, 6, 3]
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(learnable)
        # Note: size_3d is a reshape (view) of size, don't deallocate separately
        logger.debug("_kps_generator: learnable_kps done")

        # Concat fixed + learnable: [bs*num_anchor, 13, 3]
        key_points = ttnn.concat([fix_kps, learnable_kps], dim=1)
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(fix_kps)
        ttnn.deallocate(learnable_kps)
        logger.debug("_kps_generator: concat done")

        # --- Host fallback for rotation + translation (small tensors, avoids TILE hang) ---
        key_points_torch = ttnn.to_torch(key_points)  # [bs*num_anchor, 13, 3]
        ttnn.deallocate(key_points)
        anchor_torch = ttnn.to_torch(anchor)  # [bs, num_anchor, 11]

        cos_yaw = anchor_torch[:, :, COS_YAW].reshape(bs * num_anchor, 1)  # [900, 1]
        sin_yaw = anchor_torch[:, :, SIN_YAW].reshape(bs * num_anchor, 1)  # [900, 1]

        kp_x = key_points_torch[:, :, 0]  # [900, 13]
        kp_y = key_points_torch[:, :, 1]  # [900, 13]
        kp_z = key_points_torch[:, :, 2]  # [900, 13]

        rot_x = cos_yaw * kp_x - sin_yaw * kp_y
        rot_y = sin_yaw * kp_x + cos_yaw * kp_y

        key_points_torch = torch.stack([rot_x, rot_y, kp_z], dim=-1)  # [900, 13, 3]

        # Translate to anchor center
        center = anchor_torch[:, :, X:Z + 1].reshape(bs * num_anchor, 1, 3)  # [900, 1, 3]
        key_points_torch = key_points_torch + center

        # Reshape to [bs, num_anchor*num_pts, 3] and send back to device
        key_points_torch = key_points_torch.reshape(bs, num_anchor * self.num_pts, 3)
        key_points = ttnn.from_torch(
            key_points_torch.float(),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=ttnn.float32,
        )
        logger.debug("_kps_generator: rotation+translate (host) done")

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
            points_2d_grid: [bs*num_cams, num_anchor, num_pts, 2] on device
        """
        n_pts_total = num_anchor * self.num_pts

        # Append ones for homogeneous: [bs, n_pts_total, 4]
        ones = ttnn.from_torch(
            torch.ones(bs, n_pts_total, 1),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            dtype=ttnn.float32,
        )
        pts_homo = ttnn.concat([key_points, ones], dim=-1)  # [bs, n_pts_total, 4]

        # Per-camera projection via loop (avoid 6D tensor)
        all_cam_points = []
        for cam_idx in range(self.num_cams):
            logger.debug(f"DFA _project_points: cam {cam_idx}/{self.num_cams} start")
            proj = ttnn.slice(
                projection_mat,
                [0, cam_idx, 0, 0],
                [bs, cam_idx + 1, 4, 4],
            )
            proj = ttnn.reshape(proj, (bs, 4, 4))

            proj_t = ttnn.transpose(proj, -2, -1)
            projected = ttnn.matmul(pts_homo, proj_t, compute_kernel_config=self._hifi_compute_config)
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(proj)
            ttnn.deallocate(proj_t)
            logger.debug(f"DFA _project_points: cam {cam_idx} matmul done")

            xy = ttnn.slice(projected, [0, 0, 0], [bs, n_pts_total, 2])
            z = ttnn.slice(projected, [0, 0, 2], [bs, n_pts_total, 3])
            z_clamped = ttnn.clamp(z, min=1e-5)
            z_recip = ttnn.reciprocal(z_clamped)
            xy_div = ttnn.multiply(xy, z_recip)
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(projected)
            ttnn.deallocate(xy)
            ttnn.deallocate(z)
            ttnn.deallocate(z_clamped)
            ttnn.deallocate(z_recip)
            logger.debug(f"DFA _project_points: cam {cam_idx} perspective divide done")

            wh = ttnn.slice(image_wh, [0, cam_idx, 0], [bs, cam_idx + 1, 2])
            wh = ttnn.reshape(wh, (bs, 1, 2))
            wh_recip = ttnn.reciprocal(wh)
            ttnn.deallocate(wh)
            xy_norm = ttnn.multiply(xy_div, wh_recip)

            xy_scaled = ttnn.multiply(xy_norm, self._scalar_two)
            xy_grid = ttnn.subtract(xy_scaled, self._scalar_one)
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(xy_div)
            ttnn.deallocate(wh_recip)
            ttnn.deallocate(xy_norm)
            ttnn.deallocate(xy_scaled)
            logger.debug(f"DFA _project_points: cam {cam_idx} grid done")

            xy_grid = ttnn.reshape(xy_grid, (bs, num_anchor, self.num_pts, 2))
            all_cam_points.append(xy_grid)

        # Interleave camera results to match PyTorch order:
        # PyTorch: [b0_c0, b0_c1, ..., b0_c5, b1_c0, ...] (batch varies slowest)
        # Each cam result: [bs, num_anchor, num_pts, 2]
        if bs == 1:
            # Simple case: just concat cameras along dim=0
            points_2d = ttnn.concat(all_cam_points, dim=0)
        else:
            # For bs>1: need to interleave batch and camera dims
            # Slice per-batch from each camera, then concat in correct order
            interleaved = []
            for b in range(bs):
                for cam_idx in range(self.num_cams):
                    cam_data = all_cam_points[cam_idx]
                    b_slice = ttnn.slice(
                        cam_data,
                        [b, 0, 0, 0],
                        [b + 1, num_anchor, self.num_pts, 2],
                    )
                    interleaved.append(b_slice)
            points_2d = ttnn.concat(interleaved, dim=0)
            for s in interleaved:
                ttnn.deallocate(s)

        ttnn.synchronize_device(self.device)
        ttnn.deallocate(pts_homo)
        ttnn.deallocate(ones)
        for p in all_cam_points:
            ttnn.deallocate(p)

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
        ttnn.synchronize_device(self.device)
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
            ttnn.synchronize_device(self.device)
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
        ttnn.synchronize_device(self.device)

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

        # 1. grid_sample_lerp per level (lerp-based bilinear, device-only)
        all_level_features = []
        for level_idx, fm_tt in enumerate(feature_maps):
            h, w = spatial_shapes[level_idx]
            logger.debug(f"==== DFA grid_sample_lerp level {level_idx}: spatial={h}x{w}")

            # Reshape feature map: [1, 1, N*H*W, C] -> [N, H, W, C] (NHWC)
            fm = ttnn.to_memory_config(fm_tt, ttnn.DRAM_MEMORY_CONFIG)
            fm = ttnn.to_layout(fm, ttnn.ROW_MAJOR_LAYOUT)
            fm = ttnn.reshape(fm, (n_batch, h, w, self.embed_dims))

            # Grid: [bs*num_cams, num_anchor, num_pts, 2] — use float32 for precision
            grid = ttnn.to_layout(points_2d, ttnn.ROW_MAJOR_LAYOUT)
            grid = ttnn.to_memory_config(grid, ttnn.DRAM_MEMORY_CONFIG)
            if grid.dtype != ttnn.float32:
                grid = ttnn.typecast(grid, ttnn.float32)

            sampled = ttnn.grid_sample_lerp(
                fm, grid,
                padding_mode="zeros",
                align_corners=False,
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
        """Weighted fusion on device.

        Args:
            features: [bs*num_anchor, cams*levels*pts, embed_dims] on device
            weights: [bs*num_anchor, cams*levels*pts, num_groups] on device

        Returns:
            output: [bs*num_anchor, embed_dims] on device
        """
        total_clp = self.num_cams * self.num_levels * self.num_pts
        n = bs * num_anchor  # 900

        # Use matmul instead of multiply+sum(dim=1) for efficiency
        # Original: features[n,clp,G,D] * weights[n,clp,G,1] → sum(dim=1) → [n,G,D]
        # Matmul:   weights[n*G,1,clp] @ features[n*G,clp,D] → [n*G,1,D]
        logger.debug(f"fusion: matmul approach, n={n}, clp={total_clp}, groups={self.num_groups}")

        # features: [n, clp, embed_dims] → [n, clp, G, D] → [n, G, clp, D] → [n*G, clp, D]
        features = ttnn.reshape(features, (n, total_clp, self.num_groups, self.group_dims))
        features = ttnn.transpose(features, 1, 2)  # [n, G, clp, D]
        features = ttnn.reshape(features, (n * self.num_groups, total_clp, self.group_dims))

        # weights: [n, clp, G] → [n, G, clp] → [n*G, 1, clp]
        weights = ttnn.transpose(weights, -2, -1)  # [n, G, clp]
        weights = ttnn.reshape(weights, (n * self.num_groups, 1, total_clp))
        logger.debug("fusion: reshape done")

        # matmul: [n*G, 1, clp] @ [n*G, clp, D] = [n*G, 1, D]
        summed = ttnn.matmul(weights, features, compute_kernel_config=self._hifi_compute_config)
        ttnn.synchronize_device(self.device)
        logger.debug("fusion: matmul done")
        ttnn.deallocate(features)
        ttnn.deallocate(weights)

        # Reshape: [n*G, 1, D] → [n, G, D] → [1, 1, n, embed_dims]
        features = ttnn.reshape(summed, (n, self.num_groups, self.group_dims))
        features = ttnn.reshape(features, (1, 1, n, self.embed_dims))
        logger.debug("fusion: final reshape done")

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
        ttnn.synchronize_device(self.device)
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

        logger.debug("DFA run: output_proj start")
        output = ttnn.linear(
            features, self.output_proj_weight, bias=self.output_proj_bias,
            compute_kernel_config=self._hifi_compute_config,
        )
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(features)
        logger.debug("DFA run: output_proj done")

        logger.debug("DFA run: residual start")
        inst_flat = ttnn.reshape(
            instance_feature, (1, 1, bs * num_anchor, self.embed_dims)
        )
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
