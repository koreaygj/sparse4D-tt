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
# All operations on device using ttnn ops.
# anchor_projection handled on host (per-frame coordinate transform,
# requires numpy metas) then transferred back to device.
# =============================================================================

import numpy as np
import torch
import ttnn

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
        self._dev_anchor = self._to_dev(anchor_tiled)
        self._dev_feature = self._to_dev(feature_tiled)

        self.reset()

    def _to_dev(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """Helper: from_torch with mesh_mapper if mesh mode."""
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

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

            _kw = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
            if self._mesh_device is not None:
                _kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)

            anc = self.cached_anchor
            nt = anc.shape[1]
            neg_ti_bf = (-time_interval_pt).reshape(bs, 1, 1).bfloat16()
            ti_tt = ttnn.from_torch(neg_ti_bf, **_kw)
            vel = ttnn.slice(anc, [0, 0, VX], [bs, nt, VZ + 1])
            center = ttnn.slice(anc, [0, 0, X], [bs, nt, Z + 1])
            translation = ttnn.multiply(vel, ti_tt)
            center = ttnn.subtract(center, translation)
            ttnn.deallocate(translation); ttnn.deallocate(ti_tt)

            R33_T_pt = T_temp2cur_pt[:,:3,:3].transpose(-1,-2).contiguous()
            R33_T = ttnn.from_torch(R33_T_pt, **_kw)
            t3 = ttnn.from_torch(T_temp2cur_pt[:,:3,3].reshape(bs,1,3).contiguous(), **_kw)
            center = ttnn.matmul(center, R33_T)
            center = ttnn.add(center, t3); ttnn.deallocate(t3)

            cos_yaw = ttnn.slice(anc, [0,0,COS_YAW], [bs,nt,COS_YAW+1])
            sin_yaw = ttnn.slice(anc, [0,0,SIN_YAW], [bs,nt,SIN_YAW+1])
            yaw_cs = ttnn.concat([cos_yaw, sin_yaw], dim=-1)
            ttnn.deallocate(cos_yaw); ttnn.deallocate(sin_yaw)
            R22_T = ttnn.from_torch(T_temp2cur_pt[:,:2,:2].transpose(-1,-2).contiguous(), **_kw)
            yaw_out = ttnn.matmul(yaw_cs, R22_T); ttnn.deallocate(yaw_cs)

            # Reuse R33_T for velocity rotation (same matrix, was uploaded twice before)
            vel_out = ttnn.matmul(vel, R33_T)
            ttnn.deallocate(vel); ttnn.deallocate(R33_T); ttnn.deallocate(R22_T)

            size = ttnn.slice(anc, [0,0,W], [bs,nt,H+1])
            cached_anchor = ttnn.concat([center, size, yaw_out, vel_out], dim=-1)
            ttnn.deallocate(center); ttnn.deallocate(size)
            ttnn.deallocate(yaw_out); ttnn.deallocate(vel_out)
            self.cached_anchor = cached_anchor

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

        N = self.num_anchor - self.num_temp_instances  # 300

        # Device-side topk (no host roundtrip)
        conf_max = ttnn.max(confidence, dim=-1, keepdim=True)
        conf_max_2d = ttnn.reshape(conf_max, (bs, self.num_anchor))
        ttnn.deallocate(conf_max)
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
        self.cached_anchor = ttnn.gather(anchor, 1, idx_anch)
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
