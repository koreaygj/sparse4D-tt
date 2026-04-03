# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DFA Camera Parallel Runner using 2 independent submeshes.
No mesh device, no all_gather, no fabric.
Each submesh processes 3 cameras independently.
Results combined on host.
"""

import torch
import ttnn
from typing import List, Tuple
from loguru import logger

from model.deformable_feature_aggregation import DeformableFeatureAggregation


class MeshDFARunner:
    """Camera-parallel DFA using 2 submeshes (no mesh device needed)."""

    def __init__(self, dfa0: DeformableFeatureAggregation, dev1, parameters, model_config, mesh_device=None):
        """
        Args:
            dfa0: DFA instance on submesh0 (already constructed)
            dev1: submesh1 device
            parameters: DFA layer parameters (torch tensors)
            model_config: model config dict
            mesh_device: mesh device for all_gather_async (optional)
        """
        self.dfa0 = dfa0
        self.dev1 = dev1
        self.mesh_device = mesh_device

        # Create semaphores for all_gather_async if mesh device available
        if mesh_device is not None:
            core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(7,7))})
            self._ag_sem1 = ttnn.create_global_semaphore(mesh_device, core_range, 0)
            self._ag_sem2 = ttnn.create_global_semaphore(mesh_device, core_range, 0)
        self.embed_dims = dfa0.embed_dims
        self.num_groups = dfa0.num_groups
        self.group_dims = dfa0.group_dims
        self.num_levels = dfa0.num_levels
        self.num_cams_per_dev = 3
        self.num_pts = dfa0.num_pts
        self.residual_mode = dfa0.residual_mode

        # Build DFA on dev1 (same weights, different device)
        self.dfa1 = DeformableFeatureAggregation(
            device=dev1,
            parameters=parameters,
            model_config=model_config,
            embed_dims=dfa0.embed_dims,
            num_groups=dfa0.num_groups,
            num_levels=dfa0.num_levels,
            num_cams=3,  # only 3 cameras per device
            num_pts=dfa0.num_pts,
            num_learnable_pts=dfa0.num_learnable_pts,
            use_camera_embed=True,
            residual_mode="add",  # we handle residual ourselves
        )

        # Cache KPS weights on host (avoid repeated device reads)
        self._kps_fix_scale = ttnn.to_torch(dfa0.fix_scale).float().reshape(1, 7, 3)
        self._kps_lfc_w = ttnn.to_torch(dfa0.learnable_fc_weight).float()
        self._kps_lfc_b = ttnn.to_torch(dfa0.learnable_fc_bias).float().squeeze()

    def _kps_host(self, dfa, anchor_pt, inst_pt, bs, num_anchor):
        """KPS generator entirely on host (no device ops)."""
        import torch.nn.functional as F
        size = torch.exp(anchor_pt[:, :, 3:6]).reshape(bs * num_anchor, 1, 3)
        fix_kps = self._kps_fix_scale * size
        inst_flat = inst_pt.reshape(1, 1, bs * num_anchor, dfa.embed_dims)
        learnable = F.linear(inst_flat, self._kps_lfc_w.t(), self._kps_lfc_b)
        learnable = learnable.reshape(bs * num_anchor, dfa.num_learnable_pts, 3)
        learnable = torch.sigmoid(learnable) - 0.5
        learnable_kps = learnable * size

        kp = torch.cat([fix_kps, learnable_kps], dim=1)
        cos_yaw = anchor_pt[:, :, 7].reshape(bs * num_anchor, 1)
        sin_yaw = anchor_pt[:, :, 6].reshape(bs * num_anchor, 1)
        rot_x = cos_yaw * kp[:,:,0] - sin_yaw * kp[:,:,1]
        rot_y = sin_yaw * kp[:,:,0] + cos_yaw * kp[:,:,1]
        kp = torch.stack([rot_x, rot_y, kp[:,:,2]], dim=-1)
        center = anchor_pt[:, :, 0:3].reshape(bs * num_anchor, 1, 3)
        kp = (kp + center).reshape(bs, num_anchor * dfa.num_pts, 3)
        return kp

    def _project_host(self, kp_host, proj_host, wh_host, bs, num_anchor, num_pts):
        """Project 3D→2D entirely on host for all 6 cameras."""
        n_pts = num_anchor * num_pts
        ones = torch.ones(bs, n_pts, 1)
        pts_homo = torch.cat([kp_host, ones], dim=-1)  # [bs, n_pts, 4]

        all_cams = []
        for cam in range(6):
            proj = proj_host[:, cam, :, :]  # [bs, 4, 4]
            proj_t = proj.transpose(-2, -1)
            projected = torch.matmul(pts_homo, proj_t)  # [bs, n_pts, 4]
            xy = projected[:, :, :2]
            z = projected[:, :, 2:3].clamp(min=1e-5)
            xy_div = xy / z
            wh = wh_host[:, cam, :].unsqueeze(1)  # [bs, 1, 2]
            xy_norm = xy_div / wh
            xy_grid = xy_norm * 2.0 - 1.0
            xy_grid = xy_grid.reshape(bs, num_anchor, num_pts, 2)
            all_cams.append(xy_grid)

        # [6, bs, num_anchor, num_pts, 2] → [6*bs, num_anchor, num_pts, 2] for bs=1 → [6, 900, 13, 2]
        return torch.cat(all_cams, dim=0)  # [6, num_anchor, num_pts, 2]

    def _with_3cam(self, dfa, method_name, *args, **kwargs):
        """Call a DFA method with num_cams temporarily set to 3."""
        orig = dfa.num_cams
        dfa.num_cams = 3
        try:
            return getattr(dfa, method_name)(*args, **kwargs)
        finally:
            dfa.num_cams = orig

    def _get_weights_3cam(self, dfa, instance_feature, anchor_embed, proj, bs, num_anchor, return_logits=False):
        orig = dfa.num_cams
        dfa.num_cams = 3
        try:
            return dfa._get_weights(instance_feature, anchor_embed, proj, bs, num_anchor, return_logits=return_logits)
        finally:
            dfa.num_cams = orig

    def _feature_sampling_3cam(self, dfa, feature_maps, points_2d, spatial_shapes, bs, num_anchor):
        orig = dfa.num_cams
        dfa.num_cams = 3
        try:
            return dfa._feature_sampling(feature_maps, points_2d, spatial_shapes, bs, num_anchor)
        finally:
            dfa.num_cams = orig

    def _fusion_3cam(self, dfa, features, weights, bs, num_anchor):
        orig = dfa.num_cams
        dfa.num_cams = 3
        try:
            return dfa._multi_view_level_fusion(features, weights, bs, num_anchor)
        finally:
            dfa.num_cams = orig

    def _project_3cam(self, dfa, key_points, projection_mat, image_wh, bs, num_anchor):
        """Project 3D→2D for 3 cameras (reuses DFA's _project_points logic)."""
        # Temporarily override num_cams
        orig_cams = dfa.num_cams
        dfa.num_cams = 3
        try:
            result = dfa._project_points(key_points, projection_mat, image_wh, bs, num_anchor)
        finally:
            dfa.num_cams = orig_cams
        return result

    def run(self, instance_feature, anchor, anchor_embed,
            feature_maps_dev0, feature_maps_dev1,
            projection_mat, image_wh, spatial_shapes, bs=1, num_anchor=900,
            feature_maps_mesh=None):
        """
        Camera-parallel DFA: 3 cameras on each submesh.

        All non-feature_map inputs are on submesh0.
        feature_maps_dev0: list of 4 tensors on submesh0 (cam 0-2)
        feature_maps_dev1: list of 4 tensors on submesh1 (cam 3-5)
        """
        dfa0 = self.dfa0
        dfa1 = self.dfa1
        n = bs * num_anchor

        # === 1-3. KPS + Projection on HOST (avoid triple host roundtrip) ===
        # Read inputs to host once
        anchor_pt = ttnn.to_torch(anchor).float()
        inst_pt = ttnn.to_torch(instance_feature).float()
        ae_pt = ttnn.to_torch(anchor_embed).float()

        # Cache proj/wh host tensors (same within a frame)
        if not hasattr(self, '_proj_host_cache') or self._proj_host_cache is None:
            self._proj_host_cache = ttnn.to_torch(projection_mat).float()
            self._wh_host_cache = ttnn.to_torch(image_wh).float()
            # Also cache device tensors for weights computation
            proj0 = self._proj_host_cache[:, :3, :, :].contiguous()
            proj1 = self._proj_host_cache[:, 3:, :, :].contiguous()
            wh0 = self._wh_host_cache[:, :3, :].contiguous()
            wh1 = self._wh_host_cache[:, 3:, :].contiguous()
            self._proj_dev0_cache = ttnn.from_torch(proj0, layout=ttnn.TILE_LAYOUT, device=dfa0.device, dtype=ttnn.float32)
            self._proj_dev1_cache = ttnn.from_torch(proj1, layout=ttnn.TILE_LAYOUT, device=self.dev1, dtype=ttnn.float32)
            self._wh_dev0_cache = ttnn.from_torch(wh0, layout=ttnn.TILE_LAYOUT, device=dfa0.device, dtype=ttnn.float32)
            self._wh_dev1_cache = ttnn.from_torch(wh1, layout=ttnn.TILE_LAYOUT, device=self.dev1, dtype=ttnn.float32)

        # KPS on host
        kp_host = self._kps_host(dfa0, anchor_pt, inst_pt, bs, num_anchor)

        # Projection on host (3D→2D for all 6 cameras)
        pts2d_host = self._project_host(kp_host, self._proj_host_cache, self._wh_host_cache, bs, num_anchor, dfa0.num_pts)

        # Split 2D points: cam 0-2 → dev0, cam 3-5 → dev1
        pts2d_0 = pts2d_host[:3].contiguous()  # [3, 900, 13, 2]
        pts2d_1 = pts2d_host[3:].contiguous()
        points_2d_dev0 = ttnn.from_torch(pts2d_0, layout=ttnn.TILE_LAYOUT, device=dfa0.device, dtype=ttnn.float32)
        points_2d_dev1 = ttnn.from_torch(pts2d_1, layout=ttnn.TILE_LAYOUT, device=self.dev1, dtype=ttnn.float32)

        # Transfer inst/ae to dev1
        inst_dev1 = ttnn.from_torch(inst_pt, layout=ttnn.TILE_LAYOUT, device=self.dev1, dtype=ttnn.float32)
        ae_dev1 = ttnn.from_torch(ae_pt, layout=ttnn.TILE_LAYOUT, device=self.dev1, dtype=ttnn.float32)

        # === 4. Get weights on both submeshes (per-3-cam softmax on device) ===
        proj_dev0 = self._proj_dev0_cache
        proj_dev1 = self._proj_dev1_cache
        weights_dev0 = self._get_weights_3cam(dfa0, instance_feature, anchor_embed, proj_dev0, bs, num_anchor)
        weights_dev1 = self._get_weights_3cam(dfa1, inst_dev1, ae_dev1, proj_dev1, bs, num_anchor)


        # === 5-7. Sampling + Fusion + Combine ===
        if feature_maps_mesh is not None and self.mesh_device is not None and False:  # disabled
            # MESH SPMD path: grid_sample on mesh → fusion → all_gather_async
            # Create mesh grid from host points_2d
            pts2d_mesh = ttnn.from_torch(
                pts2d_host, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, dtype=ttnn.float32,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0))

            # Grid_sample on mesh SPMD (each device processes 3 cameras)
            grid = ttnn.to_layout(pts2d_mesh, ttnn.ROW_MAJOR_LAYOUT)
            grid = ttnn.to_memory_config(grid, ttnn.DRAM_MEMORY_CONFIG)
            if grid.dtype != ttnn.float32:
                grid = ttnn.typecast(grid, ttnn.float32)

            all_level = []
            for lvl, fm in enumerate(feature_maps_mesh):
                h, w = spatial_shapes[lvl]
                s = ttnn.grid_sample_lerp(fm, grid, padding_mode="zeros", align_corners=False)
                s = ttnn.to_layout(s, ttnn.TILE_LAYOUT)
                s = ttnn.to_memory_config(s, ttnn.DRAM_MEMORY_CONFIG)
                all_level.append(s)
            ttnn.deallocate(grid); ttnn.deallocate(pts2d_mesh)

            # Rearrange (same as DFA._feature_sampling but with num_cams=3 per device)
            chunks = []
            for cam in range(3):
                for lvl in range(self.num_levels):
                    chunks.append(ttnn.slice(all_level[lvl], [cam,0,0,0],
                        [cam+1, num_anchor, self.num_pts, self.embed_dims]))
            features_mesh = ttnn.concat(chunks, dim=2)
            for c in chunks: ttnn.deallocate(c)
            for s in all_level: ttnn.deallocate(s)
            features_mesh = ttnn.reshape(features_mesh, (num_anchor, 3*self.num_levels*self.num_pts, self.embed_dims))

            # Create mesh weights from submesh weights
            w0_host = ttnn.to_torch(weights_dev0).float()
            w1_host = ttnn.to_torch(weights_dev1).float()
            w_stacked = torch.cat([w0_host, w1_host], dim=0)
            weights_mesh = ttnn.from_torch(w_stacked, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                                            dtype=ttnn.float32, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0))

            # Fusion on mesh SPMD
            half_clp = 3 * self.num_levels * self.num_pts
            wf = ttnn.reshape(weights_mesh, (n * half_clp, self.num_groups))
            we = ttnn.repeat_interleave(wf, self.group_dims, dim=-1)
            ttnn.deallocate(wf); ttnn.deallocate(weights_mesh)
            we = ttnn.reshape(we, (n, half_clp, self.embed_dims))
            result = ttnn.multiply(features_mesh, we)
            ttnn.deallocate(we); ttnn.deallocate(features_mesh)
            partial = ttnn.sum(result, dim=1)
            partial = ttnn.reshape(partial, (1, 1, n, self.embed_dims))

            # all_gather_async → both devices get full result
            partial = ttnn.to_memory_config(partial, ttnn.DRAM_MEMORY_CONFIG)
            gathered = ttnn.experimental.all_gather_async(partial, dim=2,
                multi_device_global_semaphore=[self._ag_sem1, self._ag_sem2],
                num_links=1, topology=ttnn.Topology.Linear)
            ttnn.deallocate(partial)
            # gathered: [1, 1, 1800, 256] on each device. Sum halves.
            p0 = ttnn.slice(gathered, [0, 0, 0, 0], [1, 1, n, self.embed_dims])
            p1 = ttnn.slice(gathered, [0, 0, n, 0], [1, 1, 2*n, self.embed_dims])
            fused_mesh = ttnn.add(p0, p1)
            ttnn.deallocate(gathered); ttnn.deallocate(p0); ttnn.deallocate(p1)

            # Get result to submesh0
            combined_host = ttnn.to_torch(fused_mesh, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()[:1]
            combined = ttnn.from_torch(combined_host, layout=ttnn.TILE_LAYOUT, device=dfa0.device, dtype=ttnn.float32)
            ttnn.deallocate(fused_mesh)
        else:
            # Submesh path (fallback)
            features_dev0 = self._feature_sampling_3cam(dfa0, feature_maps_dev0, points_2d_dev0, spatial_shapes, bs, num_anchor)
            features_dev1 = self._feature_sampling_3cam(dfa1, feature_maps_dev1, points_2d_dev1, spatial_shapes, bs, num_anchor)
            fused_dev0 = self._fusion_3cam(dfa0, features_dev0, weights_dev0, bs, num_anchor)
            fused_dev1 = self._fusion_3cam(dfa1, features_dev1, weights_dev1, bs, num_anchor)

            # === Linearity trick: output_proj(A+B) = output_proj(A) + output_proj(B) ===
            # Dispatch output_proj on dev0 while reading dev1 result
            out_dev0 = ttnn.linear(fused_dev0, dfa0.output_proj_weight, bias=dfa0.output_proj_bias,
                                    compute_kernel_config=dfa0._hifi_compute_config)
            # Meanwhile, read dev1 fused result to host (to_torch syncs dev1)
            f1_host = ttnn.to_torch(fused_dev1).float()

        # Deallocate temporaries
        ttnn.deallocate(inst_dev1); ttnn.deallocate(ae_dev1)
        ttnn.deallocate(points_2d_dev1); ttnn.deallocate(weights_dev1)
        ttnn.deallocate(points_2d_dev0); ttnn.deallocate(weights_dev0)
        if feature_maps_mesh is None:
            try:
                ttnn.deallocate(features_dev1); ttnn.deallocate(fused_dev1)
                ttnn.deallocate(features_dev0); ttnn.deallocate(fused_dev0)
            except Exception:
                pass

        # === 8. Output projection dev1 on host + combine + residual ===
        if not hasattr(self, '_op_w_host'):
            self._op_w_host = ttnn.to_torch(dfa0.output_proj_weight).float()
            self._op_b_host = ttnn.to_torch(dfa0.output_proj_bias).float().squeeze()
        # Host output_proj for dev1 partial (small: [1,1,900,256] × [256,256])
        out1_host = torch.nn.functional.linear(f1_host, self._op_w_host.t(), self._op_b_host)
        out1_dev0 = ttnn.from_torch(out1_host.contiguous(), layout=ttnn.TILE_LAYOUT, device=dfa0.device, dtype=ttnn.float32)

        # Combine: out_dev0 already has bias, out1 needs no additional bias (bias applied once)
        # Actually: linear(A,W,b) + linear(B,W,b) = linear(A+B,W,b) + b → double bias!
        # Fix: host linear without bias, add bias once on device
        # Rewrite: linear(A,W,b) on dev0 already correct for A part.
        # For B: linear(B,W,0) on host, then add to dev0 result.
        out1_host_nobias = torch.nn.functional.linear(f1_host, self._op_w_host.t())
        out1_dev0 = ttnn.from_torch(out1_host_nobias.contiguous(), layout=ttnn.TILE_LAYOUT, device=dfa0.device, dtype=ttnn.float32)
        output = ttnn.add(out_dev0, out1_dev0)
        ttnn.deallocate(out1_dev0); ttnn.deallocate(out_dev0)

        inst_flat = ttnn.reshape(instance_feature, (1, 1, n, self.embed_dims))
        if output.dtype != inst_flat.dtype:
            output = ttnn.typecast(output, inst_flat.dtype)
        if self.residual_mode == "cat":
            output = ttnn.concat([output, inst_flat], dim=-1)
            output = ttnn.reshape(output, (bs, num_anchor, 2 * self.embed_dims))
        elif self.residual_mode == "add":
            output = ttnn.add(output, inst_flat)
            output = ttnn.reshape(output, (bs, num_anchor, self.embed_dims))

        return output
