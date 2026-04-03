# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# SparseBox3DRefinementModule for TT Devices
#
# Refines anchor boxes and predicts classification/quality scores.
#
# Forward flow:
#   1. feature = instance_feature + anchor_embed
#   2. output = refine_layers(feature)  → [bs, N, 11]
#   3. output[refine_state] += anchor[refine_state]  (residual refinement)
#   4. velocity = output[VX:] / time_interval + anchor[VX:]
#   5. cls = cls_layers(instance_feature)  → [bs, N, 10]
#   6. quality = quality_layers(feature)   → [bs, N, 2]
#
# Sparse4D config:
#   embed_dims=256, num_cls=10, refine_yaw=True, with_quality_estimation=True
#   refine_layers: (Linear→ReLU→LN)×2 → (Linear→ReLU→LN)×2 → Linear(256→11) → Scale
#   cls_layers: (Linear→ReLU→LN)×1 → (Linear→ReLU→LN)×1 → Linear(256→10)
#   quality_layers: (Linear→ReLU→LN)×1 → (Linear→ReLU→LN)×1 → Linear(256→2)
# =============================================================================

import torch
import ttnn

# Anchor box field indices
X, Y, Z = 0, 1, 2
W, L, H = 3, 4, 5
SIN_YAW, COS_YAW = 6, 7
VX, VY, VZ = 8, 9, 10


class SparseBox3DRefinementModule:
    """TT-NN implementation of SparseBox3DRefinementModule."""

    def __init__(
        self,
        device,
        parameters: dict,
        embed_dims: int = 256,
        output_dim: int = 11,
        num_cls: int = 10,
        refine_yaw: bool = True,
        with_quality_estimation: bool = True,
        mesh_device=None,
    ) -> None:
        self.device = mesh_device if mesh_device is not None else device
        self._mesh_device = mesh_device
        self.embed_dims = embed_dims
        self._hifi_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False, packer_l1_acc=False, math_approx_mode=False,
        )
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.refine_yaw = refine_yaw
        self.with_quality_estimation = with_quality_estimation

        self.refine_state = [X, Y, Z, W, L, H]
        if refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

        # Refine layers: Linear→ReLU→LN chains + final Linear + Scale
        self.refine_layers = self._load_layers(parameters["refine_layers"])
        self.refine_final_weight = self._to_device(parameters["refine_final_weight"])
        self.refine_final_bias = self._to_device_bias(parameters["refine_final_bias"])
        self.refine_scale = self._to_device_bias(parameters["refine_scale"])

        # Classification layers
        self.cls_layers = self._load_layers(parameters["cls_layers"])
        self.cls_final_weight = self._to_device(parameters["cls_final_weight"])
        self.cls_final_bias = self._to_device_bias(parameters["cls_final_bias"])

        # Quality estimation layers
        if with_quality_estimation:
            self.quality_layers = self._load_layers(parameters["quality_layers"])
            self.quality_final_weight = self._to_device(parameters["quality_final_weight"])
            self.quality_final_bias = self._to_device_bias(parameters["quality_final_bias"])

    def _load_layers(self, layer_params: list):
        loaded = []
        for p in layer_params:
            entry = {}
            entry["weight"] = self._to_device(p["weight"])
            entry["bias"] = self._to_device_bias(p["bias"])
            if "ln_weight" in p:
                entry["ln_weight"] = self._to_device_1d(p["ln_weight"])
                entry["ln_bias"] = self._to_device_1d(p["ln_bias"])
            loaded.append(entry)
        return loaded

    def _to_device(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

    def _to_device_bias(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

    def _to_device_1d(self, tensor: torch.Tensor) -> ttnn.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.reshape(1, 1, 1, -1)
        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(tensor.float(), **kwargs)

    def _run_layers(self, x: ttnn.Tensor, layers: list) -> ttnn.Tensor:
        for idx, entry in enumerate(layers):
            linear_in = x
            x = ttnn.linear(x, entry["weight"], bias=entry["bias"], compute_kernel_config=self._hifi_compute_config)
            relu_in = x
            x = ttnn.relu(x)
            if "ln_weight" in entry:
                relu_out = x
                x = ttnn.layer_norm(
                    x, weight=entry["ln_weight"], bias=entry["ln_bias"],
                    epsilon=1e-5,
                    compute_kernel_config=self._hifi_compute_config,
                )
                ttnn.deallocate(relu_in)
                ttnn.deallocate(relu_out)
            else:
                ttnn.deallocate(relu_in)
            # Deallocate previous layer output (skip idx=0: caller owns input)
            if idx > 0:
                ttnn.deallocate(linear_in)
        return x

    def run(
        self,
        instance_feature: ttnn.Tensor,
        anchor: ttnn.Tensor,
        anchor_embed: ttnn.Tensor,
        time_interval: ttnn.Tensor,
        bs: int,
        num_anchor: int,
        return_cls: bool = True,
    ):
        """Forward pass of refinement module.

        Args:
            instance_feature: [bs, num_anchor, embed_dims] on device
            anchor: [bs, num_anchor, 11] on device
            anchor_embed: [bs, num_anchor, embed_dims] on device
            time_interval: [bs] on device or scalar
            bs: batch size
            num_anchor: number of anchors
            return_cls: whether to compute classification

        Returns:
            (refined_anchor, cls, quality) - all on device
        """
        n = bs * num_anchor

        # feature = instance_feature + anchor_embed
        feature = ttnn.add(instance_feature, anchor_embed)
        feature_flat = ttnn.reshape(feature, (1, 1, n, self.embed_dims))

        # --- Refine layers ---
        refined = self._run_layers(feature_flat, self.refine_layers)
        refined_linear_in = refined
        refined = ttnn.linear(
            refined, self.refine_final_weight, bias=self.refine_final_bias,
            compute_kernel_config=self._hifi_compute_config,
        )
        ttnn.deallocate(refined_linear_in)
        # Apply Scale
        refined_pre_scale = refined
        refined = ttnn.multiply(refined, self.refine_scale)
        ttnn.deallocate(refined_pre_scale)
        refined = ttnn.reshape(refined, (bs, num_anchor, self.output_dim))

        # Residual refinement
        if self.refine_yaw:
            refined_pos = ttnn.slice(refined, [0, 0, 0], [bs, num_anchor, 8])
            anchor_pos = ttnn.slice(anchor, [0, 0, 0], [bs, num_anchor, 8])
            old_refined_pos = refined_pos
            refined_pos = ttnn.add(refined_pos, anchor_pos)
            ttnn.deallocate(old_refined_pos)
            ttnn.deallocate(anchor_pos)

            refined_vel = ttnn.slice(
                refined, [0, 0, VX], [bs, num_anchor, self.output_dim]
            )
        else:
            refined_pos = ttnn.slice(refined, [0, 0, 0], [bs, num_anchor, 6])
            anchor_pos = ttnn.slice(anchor, [0, 0, 0], [bs, num_anchor, 6])
            old_refined_pos = refined_pos
            refined_pos = ttnn.add(refined_pos, anchor_pos)
            ttnn.deallocate(old_refined_pos)
            ttnn.deallocate(anchor_pos)

            refined_yaw = ttnn.slice(refined, [0, 0, SIN_YAW], [bs, num_anchor, VX])
            refined_vel = ttnn.slice(
                refined, [0, 0, VX], [bs, num_anchor, self.output_dim]
            )
        ttnn.deallocate(refined)

        # Velocity refinement: vel = output[VX:] / time_interval + anchor[VX:]
        if self.output_dim > 8:
            ti = ttnn.reshape(time_interval, (bs, 1, 1))
            ti_recip = ttnn.reciprocal(ti)
            old_vel = refined_vel
            refined_vel = ttnn.multiply(refined_vel, ti_recip)
            ttnn.deallocate(old_vel)
            ttnn.deallocate(ti_recip)
            anchor_vel = ttnn.slice(
                anchor, [0, 0, VX], [bs, num_anchor, VX + (self.output_dim - VX)]
            )
            old_vel = refined_vel
            refined_vel = ttnn.add(refined_vel, anchor_vel)
            ttnn.deallocate(old_vel)
            ttnn.deallocate(anchor_vel)

        # Reconstruct full output
        if self.refine_yaw:
            output = ttnn.concat([refined_pos, refined_vel], dim=-1)
        else:
            output = ttnn.concat([refined_pos, refined_yaw, refined_vel], dim=-1)

        ttnn.deallocate(refined_pos)
        ttnn.deallocate(refined_vel)
        if not self.refine_yaw:
            ttnn.deallocate(refined_yaw)

        # --- Classification ---
        cls = None
        if return_cls:
            inst_flat = ttnn.reshape(
                instance_feature, (1, 1, n, self.embed_dims)
            )

            cls_feat = inst_flat
            for layer_idx, entry in enumerate(self.cls_layers):
                linear_in = cls_feat
                cls_feat = ttnn.linear(cls_feat, entry["weight"], bias=entry["bias"],
                                       compute_kernel_config=self._hifi_compute_config)
                relu_in = cls_feat
                cls_feat = ttnn.relu(cls_feat)
                if "ln_weight" in entry:
                    relu_out = cls_feat
                    cls_feat = ttnn.layer_norm(
                        cls_feat, weight=entry["ln_weight"], bias=entry["ln_bias"],
                        compute_kernel_config=self._hifi_compute_config,
                    )
            
                    ttnn.deallocate(relu_in)
                    ttnn.deallocate(relu_out)
                if layer_idx > 0:
                    ttnn.deallocate(linear_in)

            cls = ttnn.linear(
                cls_feat, self.cls_final_weight, bias=self.cls_final_bias,
                compute_kernel_config=self._hifi_compute_config,
            )
    
            ttnn.deallocate(cls_feat)

            cls = ttnn.reshape(cls, (bs, num_anchor, self.num_cls))

        # --- Quality estimation ---
        quality = None
        if return_cls and self.with_quality_estimation:
            qt_feat = self._run_layers(feature_flat, self.quality_layers)
            quality = ttnn.linear(
                qt_feat, self.quality_final_weight, bias=self.quality_final_bias,
                compute_kernel_config=self._hifi_compute_config,
            )
    
            ttnn.deallocate(qt_feat)
            quality = ttnn.reshape(quality, (bs, num_anchor, 2))

        return output, cls, quality

def _extract_linear_relu_ln_params(modules_list):
    """Extract params from list of modules: Linear→ReLU→LN chains."""
    params = []
    i = 0
    while i < len(modules_list):
        m = modules_list[i]
        if hasattr(m, 'weight') and hasattr(m, 'in_features'):
            entry = {
                "weight": m.weight.data.clone().t(),
                "bias": m.bias.data.clone(),
            }
            if i + 2 < len(modules_list) and isinstance(
                modules_list[i + 2], torch.nn.LayerNorm
            ):
                ln = modules_list[i + 2]
                entry["ln_weight"] = ln.weight.data.clone()
                entry["ln_bias"] = ln.bias.data.clone()
                i += 3
            else:
                i += 2
            params.append(entry)
        else:
            i += 1
    return params


def preprocess_refinement_parameters(pt_module) -> dict:
    """Extract parameters from SparseBox3DRefinementModule.

    Structure:
        layers: Sequential(
            Linear→ReLU→LN × (in_loops * out_loops),  # refine layers
            Linear(embed_dims, output_dim),             # final linear
            Scale([1.0] * output_dim),                  # scale
        )
        cls_layers: Sequential(Linear→ReLU→LN × ..., Linear(embed_dims, num_cls))
        quality_layers: Sequential(Linear→ReLU→LN × ..., Linear(embed_dims, 2))
    """
    params = {}

    # Refine layers
    modules = list(pt_module.layers.children())
    # Last 2 modules are the final Linear and Scale
    scale_module = modules[-1]
    final_linear = modules[-2]
    chain_modules = modules[:-2]

    params["refine_layers"] = _extract_linear_relu_ln_params(chain_modules)
    params["refine_final_weight"] = final_linear.weight.data.clone().t()
    params["refine_final_bias"] = final_linear.bias.data.clone()
    params["refine_scale"] = scale_module.scale.data.clone()

    # Classification layers
    cls_modules = list(pt_module.cls_layers.children())
    cls_final = cls_modules[-1]
    cls_chain = cls_modules[:-1]
    params["cls_layers"] = _extract_linear_relu_ln_params(cls_chain)
    params["cls_final_weight"] = cls_final.weight.data.clone().t()
    params["cls_final_bias"] = cls_final.bias.data.clone()

    # Quality layers
    if hasattr(pt_module, 'quality_layers'):
        qt_modules = list(pt_module.quality_layers.children())
        qt_final = qt_modules[-1]
        qt_chain = qt_modules[:-1]
        params["quality_layers"] = _extract_linear_relu_ln_params(qt_chain)
        params["quality_final_weight"] = qt_final.weight.data.clone().t()
        params["quality_final_bias"] = qt_final.bias.data.clone()

    return params
