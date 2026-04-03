# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Sparse4D Full Inference Pipeline (TT-NN)
#
# End-to-end flow:
#   Image [bs, 6, 3, 256, 704]
#     → ImageNet normalize (host)
#     → TtResNetBottleneck → [c2, c3, c4, c5]
#     → FPN neck → [p2, p3, p4, p5]
#     → Sparse4DHead decoder (6 layers)
#     → 3D bounding boxes
#
# Usage:
#   model = Sparse4DInference.from_pytorch(device, pt_model)
#   outputs = model.forward(images, metas)
# =============================================================================

import numpy as np
import torch
import ttnn
from loguru import logger

from model.resnet_bottleneck import create_tt_resnet_bottleneck
from model.fpn import FPN, preprocess_fpn_parameters
from model.sparse4d_head import Sparse4DHead, preprocess_sparse4d_head_parameters


# ImageNet normalization (from Sparse4D config)
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMG_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)

# Spatial shapes after ResNet stride [4, 8, 16, 32] on 256x704 input
SPATIAL_SHAPES = [(64, 176), (32, 88), (16, 44), (8, 22)]


def _dual_device_worker(device_id, images_3cam, ckpt_path, fpn_fp32, result_dict):
    """Worker process for dual-device backbone+FPN execution.

    Opens its own device independently, runs backbone+FPN on batch=3,
    stores FPN outputs (torch tensors) in shared result_dict.

    Args:
        device_id: TT device id (0 or 1)
        images_3cam: [3, 3, 256, 704] tensor (3 cameras)
        ckpt_path: checkpoint path for loading weights
        fpn_fp32: whether to use fp32 for FPN
        result_dict: multiprocessing.Manager().dict() for results
    """
    import ttnn as _ttnn
    import torchvision

    try:
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]

        device = _ttnn.open_device(device_id=device_id, l1_small_size=24576)

        # Build backbone (batch=3 → HiFi4 + fp32_acc auto-enabled)
        resnet = torchvision.models.resnet50(weights=None)
        resnet_sd = {
            k.replace("img_backbone.", ""): v
            for k, v in sd.items()
            if k.startswith("img_backbone.")
        }
        resnet.load_state_dict(resnet_sd, strict=False)
        resnet.eval()

        from model.resnet_bottleneck import create_tt_resnet_bottleneck
        backbone, _ = create_tt_resnet_bottleneck(
            torch_model=resnet, device=device, batch_size=3,
            input_height=256, input_width=704,
        )

        # Build FPN (batch=3)
        from model.fpn import FPN, _FPNParameters
        lateral_params = []
        fpn_params_list = []
        for i in range(4):
            w = sd[f"img_neck.lateral_convs.{i}.conv.weight"]
            b = sd[f"img_neck.lateral_convs.{i}.conv.bias"]
            lateral_params.append({
                "weight": _ttnn.from_torch(w),
                "bias": _ttnn.from_torch(b.reshape(1, 1, 1, -1)),
            })
            w = sd[f"img_neck.fpn_convs.{i}.conv.weight"]
            b = sd[f"img_neck.fpn_convs.{i}.conv.bias"]
            fpn_params_list.append({
                "weight": _ttnn.from_torch(w),
                "bias": _ttnn.from_torch(b.reshape(1, 1, 1, -1)),
            })

        parameters = _FPNParameters(lateral_params, fpn_params_list)
        act_dtype = _ttnn.float32 if fpn_fp32 else _ttnn.bfloat16
        w_dtype = _ttnn.float32 if fpn_fp32 else _ttnn.bfloat16
        fidelity = _ttnn.MathFidelity.HiFi4 if fpn_fp32 else _ttnn.MathFidelity.LoFi

        fpn = FPN(
            device=device, parameters=parameters, batch_size=3,
            in_channels=[256, 512, 1024, 2048], out_channels=256,
            model_config={
                "WEIGHTS_DTYPE": w_dtype,
                "ACTIVATIONS_DTYPE": act_dtype,
                "MATH_FIDELITY": fidelity,
            },
            input_spatial_shapes=SPATIAL_SHAPES,
        )

        # Run backbone
        imgs_nhwc = images_3cam.permute(0, 2, 3, 1).contiguous()
        imgs_flat = imgs_nhwc.reshape(1, 1, 3 * 256 * 704, 3)
        tt_input = _ttnn.from_torch(
            imgs_flat.float(), layout=_ttnn.ROW_MAJOR_LAYOUT,
            device=device, dtype=_ttnn.bfloat16,
        )
        backbone_features = backbone(tt_input)

        # Upcast if needed
        backbone_features = [
            _ttnn.typecast(f, _ttnn.bfloat16)
            if f.dtype != _ttnn.bfloat16 and f.dtype == _ttnn.bfloat8_b
            else f
            for f in backbone_features
        ]

        # Run FPN
        fpn_features = fpn.run(backbone_features, device)
        _ttnn.synchronize_device(device)

        # Gather to host as torch tensors
        fpn_torch = []
        for f in fpn_features:
            f_host = _ttnn.to_torch(f).float()
            fpn_torch.append(f_host)

        result_dict[device_id] = fpn_torch
        _ttnn.close_device(device)

    except Exception as e:
        result_dict[device_id] = f"ERROR: {str(e)[:300]}"
        try:
            _ttnn.close_device(device)
        except Exception:
            pass


class Sparse4DInference:
    """Full Sparse4D inference pipeline on TT device.

    Connects TtResNetBottleneck + FPN + Sparse4DHead.
    """

    def __init__(
        self,
        device,
        backbone,
        fpn,
        head: Sparse4DHead,
        num_cams: int = 6,
        embed_dims: int = 256,
        serial_cams_per_batch: int = 0,
    ) -> None:
        self.device = device
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.activation_dtype = ttnn.bfloat16
        # If > 0, run backbone+FPN in serial batches of this many cameras
        # e.g. serial_cams_per_batch=3 → two serial runs of 3 cameras each
        self.serial_cams_per_batch = serial_cams_per_batch
        # Dual-device mode: run backbone+FPN on 2 devices via separate processes
        self.dual_device_mode = False
        self._dual_ckpt_path = None
        self._dual_fpn_fp32 = False
        # Mesh parallel mode: backbone on 2 submeshes in parallel
        self.mesh_parallel_mode = False
        self._backbone_dev1 = None  # second backbone on submesh 1
        self._fpn_dev1 = None       # second FPN on submesh 1
        self._submesh0 = None
        self._submesh1 = None
        self._mesh_device = None

    @classmethod
    def from_pytorch(
        cls,
        device,
        pt_model,
        num_cams: int = 6,
        embed_dims: int = 256,
        num_decoder: int = 6,
        num_single_frame_decoder: int = 1,
        num_anchor: int = 900,
        num_temp_instances: int = 600,
        num_classes: int = 10,
        num_groups: int = 8,
        activation_dtype=ttnn.bfloat16,
    ):
        """Build TT-NN Sparse4D from a PyTorch Sparse4D model.

        Args:
            device: TT device
            pt_model: PyTorch Sparse4D model (from mmdet3d)
        """
        logger.info("Building TT-NN Sparse4D from PyTorch model...")

        # 1. Backbone (TtResNetBottleneck)
        logger.info("  Building ResNet50 backbone...")
        backbone, _ = create_tt_resnet_bottleneck(
            torch_model=pt_model.img_backbone,
            device=device,
            batch_size=num_cams,
            input_height=256,
            input_width=704,
            activation_dtype=activation_dtype,
        )

        # 2. FPN
        logger.info("  Building FPN neck...")
        fpn_params = preprocess_fpn_parameters(pt_model.img_neck)
        fpn = FPN(
            device=device,
            parameters=fpn_params,
            batch_size=num_cams,
            in_channels=[256, 512, 1024, 2048],
            out_channels=embed_dims,
            model_config={
                "WEIGHTS_DTYPE": activation_dtype,
                "ACTIVATIONS_DTYPE": ttnn.float32,
                "MATH_FIDELITY": ttnn.MathFidelity.HiFi2,
            },
            input_spatial_shapes=SPATIAL_SHAPES,
        )

        # 3. Sparse4DHead
        logger.info("  Building Sparse4DHead...")
        head_params = preprocess_sparse4d_head_parameters(pt_model.head)

        operation_order = list(pt_model.head.operation_order)

        head = Sparse4DHead(
            device=device,
            parameters=head_params,
            operation_order=operation_order,
            embed_dims=embed_dims,
            num_decoder=num_decoder,
            num_single_frame_decoder=num_single_frame_decoder,
            num_anchor=num_anchor,
            num_temp_instances=num_temp_instances,
            num_classes=num_classes,
            num_groups=num_groups,
            spatial_shapes=SPATIAL_SHAPES,
        )

        logger.info("  Build complete.")
        model = cls(device, backbone, fpn, head, num_cams, embed_dims)
        model.activation_dtype = activation_dtype
        return model

    def preprocess_images(self, images: torch.Tensor):
        """Preprocess and send images to TT device.

        Args:
            images: [bs, num_cams, 3, H, W] normalized float tensor from dataloader

        Returns:
            ttnn.Tensor on device, or list for serial batch mode
        """
        bs = images.shape[0]

        # Flatten cameras: [bs, 6, 3, 256, 704] → [bs*6, 3, 256, 704]
        imgs_flat = images.reshape(bs * self.num_cams, 3, 256, 704)

        # Convert to NHWC: [N, 3, H, W] → [N, H, W, 3]
        imgs_nhwc = imgs_flat.permute(0, 2, 3, 1).contiguous()

        if self.serial_cams_per_batch > 0:
            # Serial batch mode: return list of input tensors, one per batch
            cpb = self.serial_cams_per_batch
            num_batches = self.num_cams // cpb
            input_tensors = []
            h, w, c = imgs_nhwc.shape[1], imgs_nhwc.shape[2], imgs_nhwc.shape[3]
            for i in range(num_batches):
                chunk = imgs_nhwc[i * cpb : (i + 1) * cpb]  # [cpb, H, W, 3]
                chunk_flat = chunk.reshape(1, 1, cpb * h * w, c)
                t = ttnn.from_torch(
                    chunk_flat.float(),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    dtype=self.activation_dtype,
                )
                input_tensors.append(t)
            return input_tensors

        n, h, w, c = imgs_nhwc.shape
        imgs_flat_tt = imgs_nhwc.reshape(1, 1, n * h * w, c)
        input_tensor = ttnn.from_torch(
            imgs_flat_tt.float(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=self.activation_dtype,
        )
        return input_tensor

    def extract_features(self, input_tensor) -> list:
        """Run backbone + FPN on single device.

        Args:
            input_tensor: ttnn.Tensor or list (serial batch mode)

        Returns:
            list of FPN feature maps [p2, p3, p4, p5] on device (float32)
        """
        # Serial batch mode: run backbone+FPN on each batch, concat on host
        if self.serial_cams_per_batch > 0 and isinstance(input_tensor, list):
            return self._extract_features_serial(input_tensor)

        # Backbone: → [c2, c3, c4, c5]
        logger.debug("Running backbone...")
        backbone_features = self.backbone(input_tensor)

        # Upcast to activation_dtype before FPN (e.g. bfloat8_b → bfloat16)
        backbone_features = [
            ttnn.typecast(f, self.activation_dtype)
            if f.dtype != self.activation_dtype and f.dtype == ttnn.bfloat8_b
            else f
            for f in backbone_features
        ]

        # FPN: → [p2, p3, p4, p5]
        logger.debug("Running FPN...")
        fpn_features = self.fpn.run(backbone_features, self.device)

        # Convert to float32 for head
        fpn_features = [
            ttnn.typecast(f, ttnn.float32) if f.dtype != ttnn.float32 else f
            for f in fpn_features
        ]

        return fpn_features

    def _extract_features_serial(self, input_tensors: list) -> list:
        """Run backbone + FPN serially for each camera batch, then concat.

        Args:
            input_tensors: list of ttnn.Tensor, one per camera batch

        Returns:
            list of FPN feature maps [p2, p3, p4, p5] on device (float32)
        """
        num_batches = len(input_tensors)
        all_fpn_torch = [[] for _ in range(4)]  # 4 FPN levels

        for batch_idx, inp in enumerate(input_tensors):
            logger.debug(f"Serial backbone+FPN batch {batch_idx}/{num_batches}...")

            # Backbone
            backbone_features = self.backbone(inp)

            # Upcast to activation_dtype if needed
            backbone_features = [
                ttnn.typecast(f, self.activation_dtype)
                if f.dtype != self.activation_dtype and f.dtype == ttnn.bfloat8_b
                else f
                for f in backbone_features
            ]

            # FPN
            fpn_features = self.fpn.run(backbone_features, self.device)
            ttnn.synchronize_device(self.device)

            # Gather FPN results to host
            for level_idx, f in enumerate(fpn_features):
                f_torch = ttnn.to_torch(f).float()
                all_fpn_torch[level_idx].append(f_torch)
                ttnn.deallocate(f)

            # Deallocate backbone features
            for bf in backbone_features:
                try:
                    ttnn.deallocate(bf)
                except Exception:
                    pass

            ttnn.synchronize_device(self.device)
            logger.debug(f"Serial batch {batch_idx} done, device memory freed")

        # Concat batches on host, send to device as float32
        logger.debug("Concatenating serial FPN batches...")
        fpn_combined = []
        self._debug_fpn_torch = []  # Save for debug comparison
        for level_idx in range(4):
            # Each batch: [1, 1, cpb*H*W, 256], concat along spatial dim
            combined = torch.cat(all_fpn_torch[level_idx], dim=2)  # [1, 1, 6*H*W, 256]
            self._debug_fpn_torch.append(combined.clone())
            f_tt = ttnn.from_torch(
                combined,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                dtype=ttnn.float32,
            )
            fpn_combined.append(f_tt)
        logger.debug("Serial FPN features ready on device")

        return fpn_combined

    def forward(
        self,
        images: torch.Tensor,
        metas: dict,
        bs: int = 1,
    ) -> dict:
        """Full Sparse4D inference.

        Args:
            images: [bs, 6, 3, 256, 704] normalized images from dataloader
            metas: dict with projection_mat, image_wh, timestamp, img_metas
            bs: batch size

        Returns:
            dict with 'prediction', 'classification', 'quality'
        """
        # Mesh parallel mode: backbone+FPN on 2 submeshes, features as mesh tensors
        if self.mesh_parallel_mode:
            feature_maps = self._extract_features_mesh_parallel(images)
        elif self.dual_device_mode:
            feature_maps = self._extract_features_dual_device(images)
        else:
            input_tensor = self.preprocess_images(images)
            feature_maps = self.extract_features(input_tensor)

        # 3. Head decoder
        outputs = self.head.forward(feature_maps, metas, bs=bs)

        return outputs

    def _extract_features_dual_device(self, images: torch.Tensor) -> list:
        """Run backbone+FPN on 2 devices in parallel (batch=3 each).

        Main process: cam0-2 on device 0 (self.device, already open)
        Worker process: cam3-5 on device 1 (opened independently)

        No dispatch core conflict because each process uses a different device.
        Enables HiFi4 + fp32_dest_acc_en at batch=3 for improved precision.

        Args:
            images: [bs, 6, 3, 256, 704] normalized images

        Returns:
            list of FPN feature maps [p2, p3, p4, p5] on self.device (float32)
        """
        import multiprocessing as mp

        imgs_flat = images.reshape(self.num_cams, 3, 256, 704)
        imgs_half_0 = imgs_flat[:3]  # cam 0-2 → main (device 0)
        imgs_half_1 = imgs_flat[3:]  # cam 3-5 → worker (device 1)

        # Launch worker for device 1 (cam3-5) in background
        manager = mp.Manager()
        result_dict = manager.dict()

        worker = mp.Process(
            target=_dual_device_worker,
            args=(1, imgs_half_1, self._dual_ckpt_path, self._dual_fpn_fp32, result_dict),
        )
        worker.start()

        # Main: run cam0-2 on device 0 (self.device) simultaneously
        imgs_nhwc = imgs_half_0.permute(0, 2, 3, 1).contiguous()
        imgs_flat_tt = imgs_nhwc.reshape(1, 1, 3 * 256 * 704, 3)
        tt_input = ttnn.from_torch(
            imgs_flat_tt.float(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=self.activation_dtype,
        )

        # Backbone on device 0
        backbone_features = self.backbone(tt_input)
        backbone_features = [
            ttnn.typecast(f, self.activation_dtype)
            if f.dtype != self.activation_dtype and f.dtype == ttnn.bfloat8_b
            else f
            for f in backbone_features
        ]

        # FPN on device 0
        fpn_features_0 = self.fpn.run(backbone_features, self.device)
        ttnn.synchronize_device(self.device)

        # Gather device 0 FPN results to host
        fpn_torch_0 = []
        for f in fpn_features_0:
            fpn_torch_0.append(ttnn.to_torch(f).float())
            ttnn.deallocate(f)
        for bf in backbone_features:
            try:
                ttnn.deallocate(bf)
            except Exception:
                pass

        # Wait for worker (device 1) to finish
        worker.join(timeout=120)
        if worker.is_alive():
            worker.terminate()
            raise RuntimeError("Dual-device: device 1 timeout")

        if 1 not in result_dict:
            raise RuntimeError("Dual-device: device 1 returned no result")
        if isinstance(result_dict[1], str):
            raise RuntimeError(f"Dual-device: device 1: {result_dict[1]}")

        fpn_torch_1 = result_dict[1]  # list of 4 torch tensors

        # Concat FPN from both devices: [1,1,3*H*W,256] x 2 → [1,1,6*H*W,256]
        fpn_combined = []
        for level_idx in range(4):
            combined = torch.cat([fpn_torch_0[level_idx], fpn_torch_1[level_idx]], dim=2)
            f_tt = ttnn.from_torch(
                combined,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                dtype=ttnn.float32,
            )
            fpn_combined.append(f_tt)

        logger.debug("Dual-device FPN features combined on device 0")
        return fpn_combined

    def _extract_features_mesh_parallel(self, images: torch.Tensor) -> list:
        """Run backbone+FPN on mesh device SPMD (batch=3 per device).

        Input [6, 3, 256, 704] → ShardTensorToMesh(dim=0) → each device gets [3, 3, 256, 704]
        Backbone + FPN run identically on both devices (SPMD).
        Output: mesh tensors with 3 cameras per device (already sharded).

        Args:
            images: [bs, 6, 3, 256, 704] normalized images

        Returns:
            list of FPN feature maps as mesh tensors (sharded by camera)
        """
        imgs_flat = images.reshape(self.num_cams, 3, 256, 704)

        # NHWC + flatten: [6, 256, 704, 3] → [1, 1, 6*256*704, 3]
        # But we need to shard so each device gets 3 cams.
        # Shard as [2, 1, 3*256*704, 3] along dim=0 → each device gets [1, 1, 3*H*W, 3]
        imgs_nhwc = imgs_flat.permute(0, 2, 3, 1).contiguous()  # [6, 256, 704, 3]
        # Split into 2 halves: [3, 256, 704, 3] each, flatten to [1, 1, 3*H*W, 3]
        imgs_flat_0 = imgs_nhwc[:3].reshape(1, 1, 3 * 256 * 704, 3)
        imgs_flat_1 = imgs_nhwc[3:].reshape(1, 1, 3 * 256 * 704, 3)
        # Stack along dim=0 for ShardTensorToMesh
        imgs_stacked = torch.cat([imgs_flat_0, imgs_flat_1], dim=0)  # [2, 1, 3*H*W, 3]

        tt_input = ttnn.from_torch(
            imgs_stacked.float(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self._mesh_device,
            dtype=self.activation_dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(self._mesh_device, dim=0),
        )

        # Backbone SPMD: each device processes its 3 cameras
        logger.debug("Mesh SPMD: backbone start")
        backbone_features = self.backbone(tt_input)
        logger.debug("Mesh SPMD: backbone done")

        # Upcast if needed
        backbone_features = [
            ttnn.typecast(f, self.activation_dtype)
            if f.dtype != self.activation_dtype and f.dtype == ttnn.bfloat8_b
            else f
            for f in backbone_features
        ]

        # FPN SPMD
        logger.debug("Mesh SPMD: FPN start")
        fpn_features = self.fpn.run(backbone_features, self._mesh_device)
        logger.debug("Mesh SPMD: FPN done")

        # Deallocate backbone features
        for bf in backbone_features:
            try:
                ttnn.deallocate(bf)
            except Exception:
                pass

        # Pre-reshape for DFA grid_sample: [1,1,3*H*W,256] → [3, H, W, 256]
        for level_idx in range(4):
            h, w = SPATIAL_SHAPES[level_idx]
            f = fpn_features[level_idx]
            f = ttnn.to_memory_config(f, ttnn.DRAM_MEMORY_CONFIG)
            f = ttnn.to_layout(f, ttnn.ROW_MAJOR_LAYOUT)
            f = ttnn.reshape(f, (3, h, w, 256))
            if f.dtype != ttnn.bfloat16:
                f = ttnn.typecast(f, ttnn.bfloat16)
            fpn_features[level_idx] = f

        logger.debug("Mesh SPMD: FPN features reshaped for DFA")
        return fpn_features

    def reset(self):
        """Reset temporal state (call at start of new sequence)."""
        self.head.instance_bank.reset()

    def post_process(self, outputs: dict, score_threshold: float = 0.3):
        """Decode predictions to 3D bounding boxes.

        Args:
            outputs: dict from forward()
            score_threshold: confidence threshold

        Returns:
            list of dicts with 'boxes_3d', 'scores_3d', 'labels_3d'
        """
        # Use last decoder's output
        prediction = outputs["prediction"][-1]
        classification = outputs["classification"][-1]

        if prediction is None or classification is None:
            return []

        # To host
        if hasattr(prediction, 'dtype') and hasattr(ttnn, 'to_torch'):
            if self._mesh_device is not None:
                composer = ttnn.ConcatMeshToTensor(self._mesh_device, dim=0)
                pred = ttnn.to_torch(prediction, mesh_composer=composer).float()[:1]
                cls = ttnn.to_torch(classification, mesh_composer=composer).float()[:1]
            else:
                pred = ttnn.to_torch(prediction).float()
                cls = ttnn.to_torch(classification).float()
        else:
            pred = prediction.float()
            cls = classification.float()

        bs = pred.shape[0]
        results = []

        for b in range(bs):
            scores, labels = cls[b].sigmoid().max(dim=-1)  # [num_anchor]
            mask = scores > score_threshold

            boxes = pred[b][mask]      # [N, 11]
            scores_b = scores[mask]    # [N]
            labels_b = labels[mask]    # [N]

            results.append({
                "boxes_3d": boxes,      # [x,y,z,w,l,h,sin,cos,vx,vy,vz]
                "scores_3d": scores_b,
                "labels_3d": labels_b,
            })

        return results
