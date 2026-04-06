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
        # Tile anchor and feature across batch
        anchor_tiled = self.anchor_data.unsqueeze(0).expand(bs, -1, -1).contiguous()
        feature_tiled = self.instance_feature_data.unsqueeze(0).expand(bs, -1, -1).contiguous()

        cached_feature = None
        cached_anchor = None

        if self.cached_anchor is not None:
            # Compute time interval
            history_time = self.metas["timestamp"]
            time_interval_pt = metas["timestamp"] - history_time
            self.mask = torch.abs(time_interval_pt) <= self.max_time_interval

            # Anchor projection: transform cached anchors to current frame
            # Done on host since it requires numpy metas (T_global matrices)
            cached_anchor_pt = self._from_dev(self.cached_anchor)
            T_temp2cur = torch.tensor(
                np.stack([
                    x["T_global_inv"] @ self.metas["img_metas"][i]["T_global"]
                    for i, x in enumerate(metas["img_metas"])
                ]),
                dtype=torch.float32,
            )
            cached_anchor_pt = self._anchor_projection(
                cached_anchor_pt, T_temp2cur, -time_interval_pt
            )

            # Fix time_interval: use default where mask is False or interval is 0
            time_interval_pt = torch.where(
                torch.logical_and(time_interval_pt != 0, self.mask),
                time_interval_pt,
                torch.tensor(self.default_time_interval, dtype=time_interval_pt.dtype),
            )

            cached_anchor = self._to_dev(cached_anchor_pt)
            self.cached_anchor = cached_anchor  # update for update() to use projected version
            cached_feature = self.cached_feature  # already on device
        else:
            self.reset()
            time_interval_pt = torch.full(
                (bs,), self.default_time_interval, dtype=torch.float32
            )

        # Move to device
        instance_feature = self._to_dev(feature_tiled)
        anchor = self._to_dev(anchor_tiled)
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

        # confidence max over classes: [bs, num_anchor, num_cls] → [bs, num_anchor]
        conf_pt = self._from_dev(confidence)
        conf_max = conf_pt.max(dim=-1).values  # [bs, num_anchor]

        # topk: select top-N by confidence
        _, top_indices = torch.topk(conf_max, N, dim=1)  # [bs, N]

        # Gather selected features and anchors on host
        inst_pt = self._from_dev(instance_feature)
        anch_pt = self._from_dev(anchor)

        selected_feature = torch.gather(
            inst_pt, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.embed_dims),
        )
        selected_anchor = torch.gather(
            anch_pt, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, anch_pt.shape[-1]),
        )

        # Concat cached + selected
        cached_feat_pt = self._from_dev(self.cached_feature)
        cached_anch_pt = self._from_dev(self.cached_anchor)
        merged_feature = torch.cat([cached_feat_pt, selected_feature], dim=1)
        merged_anchor = torch.cat([cached_anch_pt, selected_anchor], dim=1)

        # Apply mask: use merged where mask=True, keep original where mask=False
        mask = self.mask[:, None, None]  # [bs, 1, 1]
        out_feature = torch.where(mask, merged_feature, inst_pt)
        out_anchor = torch.where(mask, merged_anchor, anch_pt)

        # Back to device
        instance_feature = self._to_dev(out_feature)
        anchor = self._to_dev(out_anchor)

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

        # confidence: max over classes → sigmoid
        conf_pt = self._from_dev(confidence)
        conf_scores = conf_pt.max(dim=-1).values.sigmoid()  # [bs, num_anchor]

        # Apply confidence decay to previously cached instances
        if self.confidence is not None:
            prev_conf = self._from_dev(self.confidence).squeeze()
            if prev_conf.dim() == 1:
                prev_conf = prev_conf.unsqueeze(0)
            decayed = prev_conf * self.confidence_decay
            conf_scores[:, :self.num_temp_instances] = torch.maximum(
                decayed, conf_scores[:, :self.num_temp_instances]
            )

        # topk: select top num_temp_instances
        top_conf, top_indices = torch.topk(
            conf_scores, self.num_temp_instances, dim=1
        )

        inst_pt = self._from_dev(instance_feature)
        anch_pt = self._from_dev(anchor)

        cached_feature = torch.gather(
            inst_pt, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, self.embed_dims),
        )
        cached_anchor = torch.gather(
            anch_pt, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, anch_pt.shape[-1]),
        )

        self.cached_feature = self._to_dev(cached_feature)
        self.cached_anchor = self._to_dev(cached_anchor)
        self.confidence = self._to_dev(top_conf.reshape(1, 1, bs, self.num_temp_instances))

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
