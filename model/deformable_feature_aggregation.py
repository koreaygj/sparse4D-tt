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
#   4. feature_sampling: grid_sample per FPN level (ttnn.grid_sample, device)
#      - lerp-based bilinear: 2-pass lerp reduces BF16 rounding from 2x to 1x
#   5. multi_view_level_fusion: weighted sum (ttnn.multiply + ttnn.sum, device)
#   6. output_proj: ttnn.linear + residual (device)
# =============================================================================

from typing import Dict, List, Tuple

import torch
import ttnn

# Custom kernel mode: auto-detect, or override via environment variable
# TT_CUSTOM_KERNELS=0 → force pure ttnn, TT_CUSTOM_KERNELS=1 → force custom kernels
import os as _os
_KERNELS_AVAILABLE = all(hasattr(ttnn, op) for op in ["kps_project_fused", "transposed_s2i", "grouped_weighted_sum"])
_env = _os.environ.get("TT_CUSTOM_KERNELS")
if _env is not None:
    _HAS_CUSTOM_KERNELS = _env == "1"
else:
    _HAS_CUSTOM_KERNELS = _KERNELS_AVAILABLE


# OOB compaction: drop the (camera, anchor) rows whose key points all project outside
# their image before grid_sample sees them. Roughly two thirds of rows do — with 6 cameras
# an anchor is visible to one or two. grid_sample already skips the DRAM reads for such
# points, but it still pays CB page, barrier and reduce per point, and that fixed cost is
# where its time actually goes. Bit-identical to the dense path, 103.7 -> 95.2 ms.
#
# On by default when the ops are present, like _HAS_CUSTOM_KERNELS above; TT_OOB_COMPACT=0
# forces the dense path. It needs BOTH grid_compact and the grid_sample patch that adds
# batch_index — an older tt-metal build can have the first without the second, and the
# docstring is the only thing nanobind exposes to tell them apart.
_OOB_AVAILABLE = hasattr(ttnn, "grid_compact") and "batch_index" in (ttnn.grid_sample.__doc__ or "")
_env = _os.environ.get("TT_OOB_COMPACT")
if _env is not None:
    _OOB_COMPACT = _env == "1"
else:
    _OOB_COMPACT = _HAS_CUSTOM_KERNELS and _OOB_AVAILABLE

# TT_OOB_CAP is the total row budget for the compacted grid. Shapes must not vary or every
# call recompiles, so it is fixed and has to cover the busiest frame.
#
# The rows are POOLED across the device's cameras rather than kept in one block per camera.
# That matters more than it sounds: per camera the budget must cover the busiest CAMERA,
# and the cameras do not peak together — measured over 16 scenes (11520 camera-calls,
# debug/measure_oob_capacity.py) the zero-loss budget is 3 x 563 = 1689 rows per camera
# but only 902 pooled, for exactly the same guarantee. Pooling needs grid_sample to take
# the source camera as data (its batch_index argument) instead of inferring it from a
# row's position, which is what the accompanying tt-metal change adds.
#
#   pooled per device: mean 540, p50 541, p95 738, p99 819, max 902 of 2700
#
# The default scales with the camera count, because the number of cameras a single DFA
# owns is not fixed: the mesh path gives it 3 (one device's half of 6) and the
# single-device path gives it all 6. A constant sized for 3 would silently drop about half
# the visible anchors on the single-device path.
#
# Overflow costs accuracy only, never speed: the shapes are fixed, so the kernel stops
# writing past the budget and every stage walks the same number of rows regardless.
_OOB_CAP_ENV = _os.environ.get("TT_OOB_CAP")

# COMPACT attention weights: keep them as [N, CLP*G] all the way into
# grouped_weighted_sum instead of reshaping to [N, CLP, G] and transposing to [CLP, N, G].
#
# G is 8 and a tile is 32 wide, so the [.., G] layouts spend 24 of every 32 columns on
# padding — the tensor, the transpose that follows it, and the mask multiply all move 4x
# the numbers they need to. The compact layout is what weights_fc already produces, and a
# tile of it holds 32 anchors x 4 consecutive CLP with nothing wasted.
#
# The cost is that softmax has to reduce over a stride-G axis, which no stock op does. It
# is expressed here as exp, a 0/1 matmul for the per-group sums, a second to scatter them
# back, and a divide. Measured against the reshape+softmax+transpose+mask chain it
# replaces: 1151 -> ~220 us per DFA call.
#
# Needs the grouped_weighted_sum reader that accepts [N, CLP*G]; an older build of the op
# raises on the shape rather than reading it wrong, so the failure is loud.
_env = _os.environ.get("TT_WT_COMPACT")
if _env is not None:
    _WT_COMPACT = _env == "1"
else:
    _WT_COMPACT = _HAS_CUSTOM_KERNELS

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
            math_approx_mode=False,
        )
        # The softmax denominator is a sum of 156 terms and is the one place in this path
        # where the accumulator width shows, so the 0/1 matmuls that compute and scatter it
        # run at full fidelity with an fp32 destination. They are 1-tile-wide matmuls, so
        # the extra passes cost almost nothing.
        self._exact_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        # 0/1 matrices for the compact-layout softmax. In [N, CLP*G] the CLP axis is
        # strided by G inside each row, so the per-group sum is a matmul by a matrix that
        # selects every G-th column, and putting it back is the transpose of that.
        # Both are padded to the tile width; the padding columns are zero and stay zero.
        self._clp_gather = self._clp_scatter = self._cam_block = None
        if _WT_COMPACT:
            total_clp = num_cams * num_levels * num_pts
            width = total_clp * num_groups
            gp = ((num_groups + 31) // 32) * 32
            gather = torch.zeros(width, gp)
            for c in range(total_clp):
                for g in range(num_groups):
                    gather[c * num_groups + g, g] = 1.0
            # One column block of CLP*G per camera, so a per-(camera, anchor) flag
            # broadcasts across that camera's levels, points and groups in one multiply.
            # Height is num_cams, not the padded tile height: matmul checks the LOGICAL
            # inner dimension, and the flags come in as [N, num_cams].
            block = torch.zeros(num_cams, width)
            per_cam = width // num_cams
            for c in range(num_cams):
                block[c, c * per_cam:(c + 1) * per_cam] = 1.0
            _kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            if self._mesh_device is not None:
                _kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
            self._clp_gather = ttnn.from_torch(gather, **_kw)
            self._clp_scatter = ttnn.from_torch(gather.t().contiguous(), **_kw)
            self._cam_block = ttnn.from_torch(block, **_kw)

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

        # 928 rows covers 3 cameras with the measured max of 902; scale it linearly and
        # keep it tile-aligned. TT_OOB_CAP overrides the total outright.
        self._oob_cap = (
            int(_OOB_CAP_ENV) if _OOB_CAP_ENV is not None
            else ((num_cams * 928 // 3 + 31) // 32) * 32
        )

        # Same, for the pooled compacted grid: CAP rows total instead of num_cams*900.
        # The batch-index shard must have the same height so a core's slice of it lines up
        # index-for-index with the grid sticks that core owns; its width is 8 only because
        # a sharded page has to be 32 B aligned and grid_sample reads element 0.
        _cshard_h = (self._oob_cap + 55) // 56
        self._cgrid_sharded_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                _core_grid, (_cshard_h, _padded_last), ttnn.ShardOrientation.ROW_MAJOR
            ),
        )
        self._cbidx_sharded_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_core_grid, (_cshard_h, 8), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self._cgrid = None
        self._cindex = None
        self._cflags = None
        self._cbidx = None
        self._compact_shape = None

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
        self._scalar_one = ttnn.from_torch(torch.full((1, 1, 1), 1.0), **_skw)
        self._scalar_two = ttnn.from_torch(torch.full((1, 1, 1), 2.0), **_skw)

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
        return ttnn.from_torch(tensor.bfloat16(), **kwargs)

    def _to_device_bias(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Move bias tensor to device as [1, 1, 1, N] in TILE layout."""
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.bfloat16(), **kwargs)

    def _to_device_1d(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Move 1D tensor (LayerNorm weight/bias) to device as [1, 1, 1, N]."""
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.bfloat16(), **kwargs)


    # === Fallback methods (pure ttnn ops, no custom kernels) ===

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
        ttnn.deallocate(size_wlh)

        # Reshape size for broadcasting: [bs*num_anchor, 1, 3]
        size_3d = ttnn.reshape(size, (bs * num_anchor, 1, 3))

        # Fixed key points: fix_scale [7, 3] * size [bs*num_anchor, 1, 3]
        fix_scale_3d = ttnn.reshape(self.fix_scale, (1, 7, 3))
        fix_kps = ttnn.multiply(fix_scale_3d, size_3d)  # [bs*num_anchor, 7, 3]

        # Learnable key points
        inst_flat = ttnn.reshape(
            instance_feature, (1, 1, bs * num_anchor, self.embed_dims)
        )
        learnable = ttnn.linear(
            inst_flat, self.learnable_fc_weight, bias=self.learnable_fc_bias,
            compute_kernel_config=self._hifi_compute_config,
        )  # [1, 1, bs*num_anchor, 18]

        learnable = ttnn.reshape(
            learnable, (bs * num_anchor, self.num_learnable_pts, 3)
        )  # [bs*num_anchor, 6, 3]
        learnable = ttnn.sigmoid(learnable)
        learnable = ttnn.subtract(learnable, self._scalar_half)  # sigmoid - 0.5
        learnable_kps = ttnn.multiply(learnable, size_3d)  # [bs*num_anchor, 6, 3]
        ttnn.deallocate(learnable)
        # Note: size_3d is a reshape (view) of size, don't deallocate separately

        # Concat fixed + learnable: [bs*num_anchor, 13, 3]
        key_points = ttnn.concat([fix_kps, learnable_kps], dim=1)
        ttnn.deallocate(fix_kps)
        ttnn.deallocate(learnable_kps)

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

        # Perspective divide (all cameras at once)
        xy = ttnn.slice(projected, [0, 0, 0], [bs * nc, n_pts_total, 2])
        z = ttnn.slice(projected, [0, 0, 2], [bs * nc, n_pts_total, 3])
        ttnn.deallocate(projected)
        z_clamped = ttnn.clamp(z, min=1e-5)
        z_recip = ttnn.reciprocal(z_clamped)
        xy_div = ttnn.multiply(xy, z_recip)
        ttnn.deallocate(xy); ttnn.deallocate(z); ttnn.deallocate(z_clamped); ttnn.deallocate(z_recip)

        # Grid normalization (all cameras at once)
        # image_wh: [bs, nc, 2] → [bs*nc, 1, 2]
        wh = ttnn.reshape(image_wh, (bs * nc, 1, 2))
        wh_recip = ttnn.reciprocal(wh)
        xy_norm = ttnn.multiply(xy_div, wh_recip)
        ttnn.deallocate(xy_div); ttnn.deallocate(wh_recip)

        xy_scaled = ttnn.multiply(xy_norm, self._scalar_two)
        xy_grid = ttnn.subtract(xy_scaled, self._scalar_one)
        ttnn.deallocate(xy_norm); ttnn.deallocate(xy_scaled)

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
        # The matrix is fp32 for the projection kernel's sake; this path is a matmul chain
        # whose error never reaches a pixel, so it takes the bf16 view.
        if projection_mat.dtype != ttnn.bfloat16:
            projection_mat = ttnn.typecast(projection_mat, ttnn.bfloat16)
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
        # The grid is already the Q14 fixed-point uint16 the sampler decodes; a typecast
        # here would reinterpret the bits as an unsigned integer and destroy them.

        # 1. grid_sample_lerp per level
        all_level_features = []
        for level_idx, fm_tt in enumerate(feature_maps):
            h, w = spatial_shapes[level_idx]

            # Reshape feature map: [1, 1, N*H*W, C] -> [N, H, W, C] (NHWC)
            fm = ttnn.to_memory_config(fm_tt, ttnn.DRAM_MEMORY_CONFIG)
            fm = ttnn.to_layout(fm, ttnn.ROW_MAJOR_LAYOUT)
            fm = ttnn.reshape(fm, (n_batch, h, w, self.embed_dims))

            sampled = ttnn.grid_sample(fm, grid, padding_mode="zeros", align_corners=False)
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
        # The matrix is fp32 for the projection kernel's sake; this path is a matmul chain
        # whose error never reaches a pixel, so it takes the bf16 view.
        if projection_mat.dtype != ttnn.bfloat16:
            projection_mat = ttnn.typecast(projection_mat, ttnn.bfloat16)
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
            # [1, 1, bs*N, CLP*G]. The camera came from the row axis and the linear's
            # columns are (level, point, group), so the merged column index is exactly
            # clp*G + g with clp = cam*L*K + level*K + point — the ordering the feature
            # buffer and the mask both already use.
            weights = ttnn.reshape(
                weights,
                (
                    1,
                    1,
                    bs * num_anchor,
                    self.num_cams * self.num_levels * self.num_pts * self.num_groups,
                ),
            )

        # Only the camera-embed path produces the compact layout for free: without it the
        # linear has no camera axis to fold into the columns, so CLP*G would not be the
        # row width and the reshape below is a real one either way.
        if _WT_COMPACT and self.use_camera_embed:
            if return_logits:
                return weights
            return self._softmax_clp(weights)

        if self.use_camera_embed:
            weights = ttnn.reshape(
                weights, (bs * num_anchor, total_clp, self.num_groups)
            )
        else:
            weights = ttnn.reshape(
                weights, (bs * num_anchor, total_clp, self.num_groups)
            )

        if return_logits:
            return weights  # pre-softmax logits

        if self._mesh_device is not None:
            # Same subset problem as _softmax_clp, and stock softmax gives no access to
            # its denominator, so the reduction is written out.
            weights = self._softmax_clp_dense(weights)
        else:
            weights = ttnn.softmax(
                weights,
                dim=1,
                numeric_stable=True,
                compute_kernel_config=self._hifi_compute_config,
            )

        return weights

    def _softmax_clp_dense(self, logits):
        """Softmax over dim 1 of [N, CLP, G], denominator reduced across the mesh.

        The compact path needs 0/1 matmuls because its CLP axis is strided inside the row;
        here the axis is a real one, so ttnn's own reductions do the local part and all
        that is added is the two cross-device sums. See _softmax_clp for why the shift is
        reduced as well — exp(l - m0) and exp(l - m1) cannot be added.
        """
        cc = dict(num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                  topology=ttnn.Topology.Linear)
        m = ttnn.max(logits, dim=1, keepdim=True)
        tot = ttnn.all_reduce(m, **cc)
        ttnn.deallocate(m)
        m = ttnn.multiply(tot, 1.0 / self._mesh_device.get_num_devices())
        ttnn.deallocate(tot)
        shifted = ttnn.subtract(logits, m)
        ttnn.deallocate(m)
        e = ttnn.exp(shifted)
        ttnn.deallocate(shifted)

        s = ttnn.sum(e, dim=1, keepdim=True)
        red = ttnn.all_reduce(s, **cc)
        ttnn.deallocate(s)
        w = ttnn.divide(e, red)
        ttnn.deallocate(e)
        ttnn.deallocate(red)
        return w

    def _softmax_clp(self, logits):
        """Softmax over the CLP entries while they stay strided by G inside each row.

        No stock op reduces a strided axis, so the sum is a matmul by a 0/1 matrix that
        picks every G-th column, and a second one puts the per-group denominator back
        under every column it belongs to. Everything else is elementwise on the compact
        tensor, which is a quarter the size of the [N, CLP, G] one this replaces.

        Subtracting the row maximum rather than the per-group maximum still stabilises
        the exponential — any constant does — and the per-group one would need the very
        strided reduction being avoided. It is looser by whatever spread there is between
        groups in a row; debug/probe_clp_softmax.py measures that spread against the
        exponent range so the looseness stays checkable rather than assumed.

        On a mesh the softmax is NOT local. Each device holds its own cameras, so its CLP
        axis is a subset of the one the softmax is defined over — 156 of 312 on a 3/3
        split — and normalising over the subset makes each device's weights sum to 1 on
        their own, so the all_reduce after the fusion adds two complete distributions
        instead of one. Measured, that alone is the whole of the DFA's error: replaying
        the per-device normalisation on exact reference logits reproduces PCC 0.952, and
        scoring the device against a per-device reference gives 0.999989. So the
        denominator is reduced across devices here.

        The shift has to be reduced with it. exp(l - m0) and exp(l - m1) are not summable,
        so both devices must subtract the same constant; any common constant is exact, and
        the mean of the per-device maxima is one both can reach with the Sum all_reduce
        this file already uses. It sits within half the spread of the true global maximum,
        which is what stabilisation actually needs.
        """
        m = ttnn.max(logits, dim=-1, keepdim=True)
        if self._mesh_device is not None:
            tot = ttnn.all_reduce(
                m, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )
            ttnn.deallocate(m)
            m = ttnn.multiply(tot, 1.0 / self._mesh_device.get_num_devices())
            ttnn.deallocate(tot)
        shifted = ttnn.subtract(logits, m)
        ttnn.deallocate(m)
        e = ttnn.exp(shifted)
        ttnn.deallocate(shifted)

        s = ttnn.matmul(e, self._clp_gather,
                        compute_kernel_config=self._exact_compute_config)
        if self._mesh_device is not None:
            # The gather's padding columns are zero on every device, so they stay zero.
            red = ttnn.all_reduce(
                s, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            )
            ttnn.deallocate(s)
            s = red
        sb = ttnn.matmul(s, self._clp_scatter,
                         compute_kernel_config=self._exact_compute_config)
        ttnn.deallocate(s)
        # Divide rather than multiply by a reciprocal: this is the one rounding that
        # softmax itself does, and doing it the same way keeps the two paths comparable.
        w = ttnn.divide(e, sb)
        ttnn.deallocate(e)
        ttnn.deallocate(sb)
        return w

    def _mask_weights_compact(self, weights, n, nc):
        """Zero the compact weights of the rows compaction dropped.

        Same job as _mask_weights, in the [N, CLP*G] layout: a camera owns one contiguous
        block of CLP*G/nc columns, so the per-(camera, anchor) flag becomes a mask over
        the full row via one matmul against the block matrix. Multiplying by exactly
        0.0/1.0 is exact in every float format, so the kept rows are untouched.
        """
        f = ttnn.to_layout(self._cflags, ttnn.TILE_LAYOUT)  # [nc, 1, 1, n]
        f2 = ttnn.reshape(f, (nc, n))
        ttnn.deallocate(f)
        ft = ttnn.transpose(f2, -2, -1)                     # [n, nc]
        ttnn.deallocate(f2)
        mask = ttnn.matmul(ft, self._cam_block, compute_kernel_config=self._hifi_compute_config)
        ttnn.deallocate(ft)
        out = ttnn.multiply(weights, mask)
        ttnn.deallocate(mask)
        ttnn.deallocate(weights)
        return out

    def _compact_grid(self, points_2d, n, nc, spatial_shapes):
        """Compact the sampling grid to the rows that actually hit an image.

        All cameras share one pooled list, in source-row order; each kept row records its
        camera in _cbidx, which grid_sample consumes as batch_index. Returns the sharded
        grid and batch index; the source rows and the per-anchor keep flags land in
        self._cindex / self._cflags.
        """
        # The buffers below are sized from the first call's n/nc and then reused, so a
        # later call with a different anchor count would silently write out of range.
        assert (n, nc) == (self._compact_shape or (n, nc)), (
            f"compaction buffers were built for {self._compact_shape}, got {(n, nc)}"
        )
        self._compact_shape = (n, nc)
        if self._cgrid is None:
            _kw = dict(
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if self._mesh_device is not None:
                _kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
            # Rows past a camera's kept count are never written, so seed them far out
            # of bounds: grid_sample still samples them (their index entry is SENTINEL,
            # so transposed_s2i discards the result) and must not read a stale NaN.
            # Q14 fixed point in a uint16 container, matching kps_project_fused. The seed
            # is the saturated value, which decodes to just under 2.0 — far outside the
            # bounds test, so a slot that is never written can only ever be discarded.
            self._cgrid = ttnn.from_torch(
                torch.full((1, self._oob_cap, 1, self.num_pts * 2), 32767, dtype=torch.int32),
                dtype=ttnn.uint16, **_kw,
            )
            self._cindex = ttnn.from_torch(
                torch.zeros(1, 1, 1, self._oob_cap, dtype=torch.int32),
                dtype=ttnn.uint32, **_kw,
            )
            self._cflags = ttnn.from_torch(
                torch.zeros(nc, 1, 1, n), dtype=ttnn.bfloat16, **_kw
            )
            self._cbidx = ttnn.from_torch(
                torch.zeros(1, 1, self._oob_cap, 8, dtype=torch.int32),
                dtype=ttnn.uint32, **_kw,
            )

        # With align_corners=False a point contributes iff g is in [-1 - 1/S, 1 + 1/S),
        # S being W on x and H on y. Take the loosest LEVEL so one compaction serves all
        # four — the grid itself is level-independent — but keep the axes separate: the
        # coarsest level is 8 x 22, so sharing 1 + 1/8 across both would keep everything.
        thr_x = 1.0 + 1.0 / min(w for _, w in spatial_shapes)
        thr_y = 1.0 + 1.0 / min(h for h, _ in spatial_shapes)
        ttnn.grid_compact(
            points_2d, self._cgrid, self._cindex, self._cflags,
            num_pts=self.num_pts, capacity=self._oob_cap, anchors=n,
            threshold_x=thr_x, threshold_y=thr_y, bidx=self._cbidx,
        )
        return (
            ttnn.to_memory_config(self._cgrid, self._cgrid_sharded_mem),
            ttnn.to_memory_config(self._cbidx, self._cbidx_sharded_mem),
        )

    def _mask_weights(self, weights_t, n, nc):
        """Zero the attention weights of the rows compaction dropped.

        A dropped row is never written into _rearrange_buf, which therefore still holds
        LAST frame's features in that slot; grouped_weighted_sum would multiply that
        stale data by a live weight. Masking annihilates it, and multiplying by exactly
        0.0/1.0 is exact in every float format, so the kept rows are untouched.
        Zeroing the 68.6 MB feature buffer instead would cost more than compaction saves.
        """
        f = ttnn.to_layout(self._cflags, ttnn.TILE_LAYOUT)  # [nc, 1, 1, n]
        mask = ttnn.transpose(f, -2, -1)  # [nc, 1, n, 1]
        ttnn.deallocate(f)
        # clp = cam*NL*K + level*K + pt, so the camera axis splits off the front of
        # weights_t for free (leading dims carry no tile padding) and the flag then
        # broadcasts over the level/point axis and the group axis in one multiply.
        w4 = ttnn.reshape(
            weights_t, (nc, self.num_levels * self.num_pts, n, self.num_groups)
        )
        out = ttnn.multiply(w4, mask)
        ttnn.deallocate(mask)
        ttnn.deallocate(w4)
        return ttnn.reshape(
            out, (nc * self.num_levels * self.num_pts, n, self.num_groups)
        )

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
        # Key points stay bf16: the kernel reads them as bf16 pages, and they are OFFSETS
        # of object size (1-5 m) rather than positions, so their absolute error is an order
        # of magnitude below the centre's. Only the anchor keeps the extra width.
        anchor_bf16 = anchor
        if anchor.dtype != ttnn.bfloat16:
            anchor_bf16 = ttnn.typecast(anchor, ttnn.bfloat16)
        key_points = self._kps_generator_pre_rotation(
            anchor_bf16, instance_feature, bs, num_anchor
        )
        if anchor_bf16 is not anchor:
            ttnn.deallocate(anchor_bf16)
        # the kernel reads the anchor as row-major pages
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

        if _HAS_CUSTOM_KERNELS:
            # === Fast path: custom kernels ===
            points_2d = ttnn.kps_project_fused(
                key_points, anchor_rm,
                self._cached_proj_rm, self._cached_wh_rm,
                num_cams=nc, num_pts=self.num_pts,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(key_points)
            ttnn.deallocate(anchor_rm)

            # Allocated before anything compaction-specific so its DRAM address does not
            # depend on TT_OOB_COMPACT, which keeps an A/B against the dense path honest.
            total_clp = nc * self.num_levels * self.num_pts
            if getattr(self, "_rearrange_buf", None) is None:
                _buf = torch.zeros(total_clp, n, self.embed_dims, dtype=torch.bfloat16)
                _kw = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
                if self._mesh_device is not None:
                    _kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
                self._rearrange_buf = ttnn.from_torch(_buf, **_kw)

            bidx_sh = None
            if _OOB_COMPACT:
                points_2d_sh, bidx_sh = self._compact_grid(
                    points_2d, n, nc, spatial_shapes
                )
            else:
                points_2d_sh = ttnn.to_memory_config(points_2d, self._grid_sharded_mem)
            ttnn.deallocate(points_2d)

            if _WT_COMPACT and self.use_camera_embed:
                # Already [1, 1, N, CLP*G] — the layout grouped_weighted_sum reads
                # directly, so neither the reshape into [N, CLP, G] nor the transpose
                # that followed it happens at all.
                weights_t = weights
                if _OOB_COMPACT:
                    weights_t = self._mask_weights_compact(weights_t, n, nc)
            else:
                weights_t = ttnn.transpose(weights, 0, 1)
                if _OOB_COMPACT:
                    weights_t = self._mask_weights(weights_t, n, nc)

            for level_idx, fm_tt in enumerate(feature_maps):
                sampled = ttnn.grid_sample(
                    fm_tt, points_2d_sh, padding_mode="zeros", align_corners=False,
                    batch_index=bidx_sh,
                )
                ttnn.transposed_s2i(
                    sampled, self._rearrange_buf,
                    num_cams=nc, num_pts=self.num_pts, num_anchors=n,
                    num_levels=self.num_levels, level=level_idx,
                    index=self._cindex if _OOB_COMPACT else None,
                    capacity=self._oob_cap if _OOB_COMPACT else 0,
                )
            ttnn.deallocate(points_2d_sh)
            if bidx_sh is not None:
                ttnn.deallocate(bidx_sh)

            # Feed the ROW_MAJOR buffer straight to grouped_weighted_sum: its RM_MODE
            # tilizes in L1, so the ttnn.to_layout(..., TILE_LAYOUT) this replaces — a
            # 137 MB DRAM round trip per call — is not needed. RM_MODE is only correct
            # with the kernel fix that configures both compute pipelines before the loop;
            # without it the op is wrong on its first call after any other op has run.
            gws_out = ttnn.grouped_weighted_sum(
                self._rearrange_buf, weights_t,
                num_groups=self.num_groups, group_dims=self.group_dims,
            )
            ttnn.deallocate(weights_t)
            n_padded = ((n + 31) // 32) * 32
            chunk0 = ttnn.slice(gws_out, [0, 0], [n, self.embed_dims])
            chunk1 = ttnn.slice(gws_out, [n_padded, 0], [n_padded + n, self.embed_dims])
            ttnn.deallocate(gws_out)
            features = ttnn.add(chunk0, chunk1)
            ttnn.deallocate(chunk0)
            ttnn.deallocate(chunk1)
        else:
            # === Fallback path: original pure ttnn ops (from pre-kernel commit bc5e388) ===
            ttnn.deallocate(anchor_rm)

            # 1. KPS generator (device-side rotation + translation)
            key_points_full = self._kps_generator(anchor, instance_feature, bs, num_anchor)

            # 2. Project to 2D per camera (device-side batched matmul)
            points_2d = self._project_points(
                key_points_full, projection_mat, image_wh, bs, num_anchor
            )
            ttnn.deallocate(key_points_full)
            ttnn.deallocate(key_points)

            # 3. Feature sampling (grid_sample per level + rearrange)
            features = self._feature_sampling(
                feature_maps, points_2d, spatial_shapes, bs, num_anchor
            )

            # 4. Weighted fusion (repeat_interleave + multiply + sum)
            features = self._multi_view_level_fusion(features, weights, bs, num_anchor)
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
