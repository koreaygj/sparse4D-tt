# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# InstanceBank for TT Devices (Inference Only)
#
# Manages anchor instances across frames for temporal modeling.
# Inference-only: no denoising queries, no gradient, no training logic.
#
# Per-frame flow:
#   get()    → prepare anchor/feature, apply temporal projection
#   update() → merge cached + new instances via topk + mask
#   cache()  → save top-K instances for next frame
#
# All operations on device using ttnn ops. The temporal anchor projection is
# folded into a single matmul+add against an 11x11 affine built on host from
# the frame's ego-pose metas — host touches the metas (their birthplace), the
# device does all the anchor math.
# =============================================================================

import numpy as np
import torch
import ttnn

import os as _os

# TT_ANCHOR_FP32=0 puts the anchor back in bf16 for an A/B.
_ANCHOR_DTYPE = ttnn.bfloat16 if _os.environ.get("TT_ANCHOR_FP32") == "0" else ttnn.float32

# Custom top-k kernel. ttnn.topk tile-pads the [1, 900] confidence row to
# 32 x 928 and bitonic-sorts all 32 rows on one core (1.9 + 2.8 ms/frame between
# the two calls); topk_select radix-sorts the one real row. TT_TOPK_KERNEL=0
# falls back to ttnn.topk. Tie order differs from ttnn.topk (ours is torch's
# lower-index-first), so an A/B against the fallback is not bit-identical on
# tied confidences.
_TOPK_KERNEL = (
    hasattr(ttnn, "topk_select") and _os.environ.get("TT_TOPK_KERNEL", "1") == "1"
)

# Anchor box field indices
X, Y, Z = 0, 1, 2
W, L, H = 3, 4, 5
SIN_YAW, COS_YAW = 6, 7
VX, VY, VZ = 8, 9, 10


class InstanceBank:
    """TT-NN InstanceBank for inference.

    Manages temporal instance caching and retrieval on device.
    """

    def __init__(
        self,
        device,
        anchor_data: torch.Tensor,
        instance_feature_data: torch.Tensor,
        num_anchor: int = 900,
        embed_dims: int = 256,
        num_temp_instances: int = 600,
        default_time_interval: float = 0.5,
        confidence_decay: float = 0.6,
        max_time_interval: float = 2.0,
        mesh_device=None,
    ) -> None:
        self.device = mesh_device if mesh_device is not None else device
        self._mesh_device = mesh_device
        self.num_anchor = num_anchor
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        # Learnable parameters (loaded from checkpoint)
        self.anchor_data = anchor_data.float()  # [num_anchor, 11] on host
        self.instance_feature_data = instance_feature_data.float()  # [num_anchor, embed_dims]

        # Pre-cache constant anchor/feature on device (avoid repeated host→device upload)
        anchor_tiled = anchor_data.float().unsqueeze(0).contiguous()  # [1, num_anchor, 11]
        feature_tiled = instance_feature_data.float().unsqueeze(0).contiguous()  # [1, num_anchor, embed_dims]
        # The anchor is fp32, not bf16, and it is the one tensor in the head where that is
        # worth it. Its x and y reach +-60 m, so bf16's relative error becomes 0.03 m of
        # absolute error, and the projection divides by depth — which cancels the distance
        # and leaves a flat 0.081 px of grid error on the finest FPN level (measured).
        # It is also [900, 11], so fp32 costs 20 KB a frame.
        self._dev_anchor = self._to_dev(anchor_tiled, _ANCHOR_DTYPE)
        self._dev_feature = self._to_dev(feature_tiled)

        # Persistent output buffers for topk_select, one (values, indices) pair
        # per k. The op writes every slot on every call, so reuse is safe.
        self._topk_bufs = {}

        self.reset()

    def _topk_out(self, k: int):
        buf = self._topk_bufs.get(k)
        if buf is None:
            kw = dict(device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
            if self._mesh_device is not None:
                kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
            vals = ttnn.from_torch(
                torch.zeros(1, 1, 1, k), dtype=ttnn.bfloat16, **kw)
            idxs = ttnn.from_torch(
                torch.zeros(1, 1, 1, k, dtype=torch.int32), dtype=ttnn.uint32, **kw)
            buf = (vals, idxs)
            self._topk_bufs[k] = buf
        return buf

    def _to_dev(self, tensor: torch.Tensor, dtype=None) -> ttnn.Tensor:
        """Helper: from_torch with mesh_mapper if mesh mode."""
        dtype = dtype or ttnn.bfloat16
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=dtype)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        host = tensor.float() if dtype == ttnn.float32 else tensor.bfloat16()
        return ttnn.from_torch(host, **kwargs)

    def _from_dev(self, tensor: ttnn.Tensor) -> torch.Tensor:
        """Helper: to_torch with mesh_composer if mesh mode."""
        if self._mesh_device is not None:
            return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(self._mesh_device, dim=0)).float()[:1]
        return ttnn.to_torch(tensor).float()

    def reset(self):
        """Reset temporal cache."""
        self.cached_feature = None  # ttnn tensor or None
        self.cached_anchor = None  # ttnn tensor or None
        self.metas = None
        self.mask = None  # torch tensor [bs] bool
        self.confidence = None  # ttnn tensor or None
        self.instance_id = None
        self.prev_id = 0

    def get(self, bs: int, metas: dict):
        """Prepare instances for current frame.

        Args:
            bs: batch size
            metas: dict with 'timestamp', 'img_metas' (containing T_global, T_global_inv)

        Returns:
            instance_feature: [bs, num_anchor, embed_dims] on device
            anchor: [bs, num_anchor, 11] on device
            cached_feature: [bs, num_temp, embed_dims] on device or None
            cached_anchor: [bs, num_temp, 11] on device or None
            time_interval: [bs] on device
        """
        # Use pre-cached device tensors (no host→device upload needed)

        cached_feature = None
        cached_anchor = None

        if self.cached_anchor is not None:
            # Compute time interval
            history_time = self.metas["timestamp"]
            time_interval_pt = metas["timestamp"] - history_time
            self.mask = torch.abs(time_interval_pt) <= self.max_time_interval

            # Anchor projection: device-side (PCC 0.999994)
            T_temp2cur_pt = torch.tensor(
                np.stack([
                    x["T_global_inv"] @ self.metas["img_metas"][i]["T_global"]
                    for i, x in enumerate(metas["img_metas"])
                ]),
                dtype=torch.float32,
            ).bfloat16()

            # The rotation matrices multiply the anchor, so they follow its dtype — a
            # mixed-dtype matmul is not what we want here and ttnn would reject it anyway.
            _kw = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=_ANCHOR_DTYPE)
            if self._mesh_device is not None:
                _kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)

            # Every projected field is LINEAR in the anchor fields:
            #   center' = (center - ti_used*vel) @ R^T + t
            #   size'   = size
            #   yaw'    = [cos, sin] @ R22^T   (written back [cos', sin'] into the
            #             (SIN, COS) slots — the upstream Sparse4D swap, preserved
            #             for checkpoint compatibility)
            #   vel'    = vel @ R^T
            # so the whole projection folds into ONE anchor @ A + b. The previous
            # form spelled it out as 13 device ops with 4 h2d uploads per frame;
            # host dispatch is the frame's scarce resource and a trace capture
            # needs the op stream minimal and static, so A and b (two tiny
            # constants) are built on host instead. ti_used = -dt, matching the
            # original InstanceBank's anchor_projection(-time_interval) call.
            Tm = T_temp2cur_pt.float()           # bf16-rounded, as uploaded before
            RT = Tm[:, :3, :3].transpose(-1, -2)
            ti_used = (-time_interval_pt).bfloat16().float().reshape(bs, 1, 1)
            A = torch.zeros(bs, 11, 11)
            b = torch.zeros(bs, 1, 11)
            A[:, X : Z + 1, X : Z + 1] = RT
            A[:, VX:, X : Z + 1] = -ti_used * RT
            b[:, 0, X : Z + 1] = Tm[:, :3, 3]
            for d in (W, L, H):
                A[:, d, d] = 1.0
            R22T = Tm[:, :2, :2].transpose(-1, -2)
            A[:, COS_YAW, SIN_YAW] = R22T[:, 0, 0]  # col SIN receives cos'
            A[:, SIN_YAW, SIN_YAW] = R22T[:, 1, 0]
            A[:, COS_YAW, COS_YAW] = R22T[:, 0, 1]  # col COS receives sin'
            A[:, SIN_YAW, COS_YAW] = R22T[:, 1, 1]
            A[:, VX:, VX:] = RT

            A_tt = ttnn.from_torch(A, **_kw)
            b_tt = ttnn.from_torch(b, **_kw)
            cached_anchor = ttnn.matmul(self.cached_anchor, A_tt)
            cached_anchor = ttnn.add(cached_anchor, b_tt)
            ttnn.deallocate(A_tt); ttnn.deallocate(b_tt)
            self.cached_anchor = self._as_bank_dtype(cached_anchor)

            time_interval_pt = torch.where(
                torch.logical_and(time_interval_pt != 0, self.mask),
                time_interval_pt,
                torch.tensor(self.default_time_interval, dtype=time_interval_pt.dtype),
            )
            cached_feature = self.cached_feature  # already on device
        else:
            self.reset()
            time_interval_pt = torch.full(
                (bs,), self.default_time_interval, dtype=torch.float32
            )

        # Use pre-cached device tensors (no host→device upload for constant data)
        instance_feature = self._dev_feature
        anchor = self._dev_anchor
        time_interval = self._to_dev(time_interval_pt.reshape(1, 1, 1, bs))
        time_interval = ttnn.reshape(time_interval, (bs,))

        return instance_feature, anchor, cached_feature, cached_anchor, time_interval

    def _as_bank_dtype(self, anchor: ttnn.Tensor) -> ttnn.Tensor:
        """The bank stores anchors in one dtype, whatever route they arrived by.

        The anchor reaches update()/cache() from the refinement, and reaches the temporal
        path from a chain of slices and matmuls; keeping the invariant here rather than
        chasing every producer is what stops a concat between the two from failing on a
        dtype mismatch. Widening a bf16 value costs nothing — it is already rounded.
        """
        if anchor.dtype != _ANCHOR_DTYPE:
            return ttnn.typecast(anchor, _ANCHOR_DTYPE)
        return anchor

    def update(
        self,
        instance_feature: ttnn.Tensor,
        anchor: ttnn.Tensor,
        confidence: ttnn.Tensor,
        bs: int,
    ):
        """Merge cached and new instances based on confidence.

        Args:
            instance_feature: [bs, num_anchor, embed_dims] on device
            anchor: [bs, num_anchor, 11] on device
            confidence: [bs, num_anchor, num_cls] on device
            bs: batch size

        Returns:
            instance_feature: [bs, num_anchor, embed_dims] on device
            anchor: [bs, num_anchor, 11] on device
        """
        if self.cached_feature is None:
            return instance_feature, anchor

        anchor = self._as_bank_dtype(anchor)
        N = self.num_anchor - self.num_temp_instances  # 300

        # Device-side topk (no host roundtrip)
        conf_max = ttnn.max(confidence, dim=-1, keepdim=True)
        conf_max_2d = ttnn.reshape(conf_max, (bs, self.num_anchor))
        ttnn.deallocate(conf_max)
        if _TOPK_KERNEL:
            val_buf, idx_buf = self._topk_out(N)
            ttnn.topk_select(conf_max_2d, val_buf, idx_buf, N)
            ttnn.deallocate(conf_max_2d)
            # Persistent buffer: reshape to a fresh view, never deallocated here.
            top_idx = ttnn.reshape(idx_buf, (bs, N, 1))
            top_idx = ttnn.to_layout(top_idx, ttnn.TILE_LAYOUT)
        else:
            _, top_idx_flat = ttnn.topk(conf_max_2d, N, dim=-1)
            ttnn.deallocate(conf_max_2d)
            top_idx_flat = ttnn.typecast(top_idx_flat, ttnn.uint32)
            top_idx = ttnn.reshape(top_idx_flat, (bs, N, 1))
            top_idx = ttnn.to_layout(top_idx, ttnn.TILE_LAYOUT)
            ttnn.deallocate(top_idx_flat)

        idx_feat = ttnn.repeat_interleave(top_idx, self.embed_dims, dim=-1)
        selected_feature = ttnn.gather(instance_feature, 1, idx_feat)
        ttnn.deallocate(idx_feat)

        anch_dim = anchor.shape[-1]
        idx_anch = ttnn.repeat_interleave(top_idx, anch_dim, dim=-1)
        selected_anchor = ttnn.gather(anchor, 1, idx_anch)
        ttnn.deallocate(idx_anch); ttnn.deallocate(top_idx)

        # Device concat
        merged_feature = ttnn.concat([self.cached_feature, selected_feature], dim=1)
        merged_anchor = ttnn.concat([self.cached_anchor, selected_anchor], dim=1)
        ttnn.deallocate(selected_feature); ttnn.deallocate(selected_anchor)

        if self.mask.all():
            instance_feature = merged_feature
            anchor = merged_anchor
        else:
            inst_pt = self._from_dev(instance_feature)
            anch_pt = self._from_dev(anchor)
            merged_f_pt = self._from_dev(merged_feature)
            merged_a_pt = self._from_dev(merged_anchor)
            mask_t = self.mask[:, None, None]
            instance_feature = self._to_dev(torch.where(mask_t, merged_f_pt, inst_pt))
            anchor = self._to_dev(torch.where(mask_t, merged_a_pt, anch_pt))
            ttnn.deallocate(merged_feature); ttnn.deallocate(merged_anchor)

        return instance_feature, anchor

    def cache(
        self,
        instance_feature: ttnn.Tensor,
        anchor: ttnn.Tensor,
        confidence: ttnn.Tensor,
        metas: dict,
        bs: int,
    ):
        """Cache top temporal instances for next frame.

        Args:
            instance_feature: [bs, num_anchor, embed_dims] on device
            anchor: [bs, num_anchor, 11] on device
            confidence: [bs, num_anchor, num_cls] on device
            metas: current frame metadata
            bs: batch size
        """
        if self.num_temp_instances <= 0:
            return

        self.metas = metas

        # confidence: max over classes → sigmoid (device-side)
        conf_max = ttnn.max(confidence, dim=-1, keepdim=True)  # [bs, num_anchor, 1]
        conf_max = ttnn.reshape(conf_max, (bs, self.num_anchor))
        conf_scores = ttnn.sigmoid(conf_max)  # [bs, num_anchor]
        ttnn.deallocate(conf_max)

        # Apply confidence decay to previously cached instances (device-side)
        if self.confidence is not None:
            prev_conf = ttnn.reshape(self.confidence, (bs, self.num_temp_instances))
            _kw_sc = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
            if self._mesh_device is not None:
                _kw_sc["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
            decay_scalar = ttnn.from_torch(
                torch.full((1, 1), self.confidence_decay), **_kw_sc)
            decayed = ttnn.multiply(prev_conf, decay_scalar)
            # Slice first num_temp from conf_scores, max with decayed, replace
            conf_temp = ttnn.slice(conf_scores, [0, 0], [bs, self.num_temp_instances])
            conf_temp = ttnn.maximum(decayed, conf_temp)
            ttnn.deallocate(decayed)
            # Reconstruct: [decayed_temp | rest]
            if self.num_temp_instances < self.num_anchor:
                conf_rest = ttnn.slice(conf_scores, [0, self.num_temp_instances],
                                        [bs, self.num_anchor])
                conf_scores = ttnn.concat([conf_temp, conf_rest], dim=-1)
                ttnn.deallocate(conf_rest)
            else:
                conf_scores = conf_temp
            ttnn.deallocate(conf_temp)

        # Device topk + gather
        K = self.num_temp_instances
        if _TOPK_KERNEL:
            val_buf, idx_buf = self._topk_out(K)
            ttnn.topk_select(conf_scores, val_buf, idx_buf, K)
            ttnn.deallocate(conf_scores)
            # Downstream (decay multiply next frame) works on TILE.
            top_conf = ttnn.to_layout(val_buf, ttnn.TILE_LAYOUT)
            top_idx = ttnn.reshape(idx_buf, (bs, K, 1))
            top_idx = ttnn.to_layout(top_idx, ttnn.TILE_LAYOUT)
        else:
            top_conf, top_idx_flat = ttnn.topk(conf_scores, K, dim=-1)
            ttnn.deallocate(conf_scores)
            top_idx_flat = ttnn.typecast(top_idx_flat, ttnn.uint32)
            top_idx = ttnn.reshape(top_idx_flat, (bs, K, 1))
            top_idx = ttnn.to_layout(top_idx, ttnn.TILE_LAYOUT)
            ttnn.deallocate(top_idx_flat)

        idx_feat = ttnn.repeat_interleave(top_idx, self.embed_dims, dim=-1)
        self.cached_feature = ttnn.gather(instance_feature, 1, idx_feat)
        ttnn.deallocate(idx_feat)

        anch_dim = anchor.shape[-1]
        idx_anch = ttnn.repeat_interleave(top_idx, anch_dim, dim=-1)
        self.cached_anchor = ttnn.gather(self._as_bank_dtype(anchor), 1, idx_anch)
        ttnn.deallocate(idx_anch); ttnn.deallocate(top_idx)

        self.confidence = ttnn.reshape(top_conf, (1, 1, bs, K))

    @staticmethod
    def _anchor_projection(
        anchor: torch.Tensor,
        T_src2dst: torch.Tensor,
        time_interval: torch.Tensor,
    ) -> torch.Tensor:
        """Project cached anchors to current frame coordinates.

        Args:
            anchor: [bs, num_temp, 11]
            T_src2dst: [bs, 4, 4] transformation matrix
            time_interval: [bs] time delta (negative = backward)

        Returns:
            projected anchor: [bs, num_temp, 11]
        """
        vel = anchor[..., VX:]
        vel_dim = vel.shape[-1]
        T = T_src2dst.unsqueeze(1).to(dtype=anchor.dtype)

        center = anchor[..., [X, Y, Z]]

        # Adjust center by velocity * time
        if time_interval is not None:
            ti = time_interval.to(dtype=vel.dtype)
            translation = vel.transpose(0, -1) * ti
            translation = translation.transpose(0, -1)
            center = center - translation

        # Rotate center
        center = (
            torch.matmul(T[..., :3, :3], center[..., None]).squeeze(-1)
            + T[..., :3, 3]
        )

        size = anchor[..., [W, L, H]]

        # Rotate yaw: matmul input is [cos, sin], output is [cos', sin']
        # Note: result order is [cos', sin'] but anchor convention is SIN=6, COS=7.
        # This is a known issue in the original Sparse4D (TODO: Fix bug comment).
        # Kept as-is for compatibility with pretrained weights.
        yaw = torch.matmul(
            T[..., :2, :2],
            anchor[..., [COS_YAW, SIN_YAW], None],
        ).squeeze(-1)

        # Rotate velocity
        vel = torch.matmul(
            T[..., :vel_dim, :vel_dim], vel[..., None]
        ).squeeze(-1)

        return torch.cat([center, size, yaw, vel], dim=-1)


def preprocess_instance_bank_parameters(pt_bank) -> dict:
    """Extract parameters from PyTorch InstanceBank.

    Args:
        pt_bank: PyTorch InstanceBank instance

    Returns:
        dict with anchor_data and instance_feature_data
    """
    return {
        "anchor_data": pt_bank.anchor.data.clone(),
        "instance_feature_data": pt_bank.instance_feature.data.clone(),
    }
