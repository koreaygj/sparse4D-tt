# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# SparseBox3DEncoder for TT Devices
#
# Encodes 3D anchor box parameters into feature embeddings.
# Each component (pos, size, yaw, vel) passes through its own
# Linear→ReLU→LN chain, then combined via "cat" mode.
#
# Sparse4D config (decouple_attn=True):
#   embed_dims = [128, 32, 32, 64]
#   mode = "cat" → output = cat([pos, size, yaw, vel]) = 256
#   output_fc = None (no final FC)
#   in_loops = 1, out_loops = 4
#
# Each embedding_layer = (Linear→ReLU→LN) × out_loops:
#   e.g. pos_fc: Linear(3→128)→ReLU→LN → Linear(128→128)→ReLU→LN × 3
# =============================================================================

import torch
import ttnn

# Anchor box field indices
X, Y, Z = 0, 1, 2
W, L, H = 3, 4, 5
SIN_YAW, COS_YAW = 6, 7
VX, VY, VZ = 8, 9, 10


class SparseBox3DEncoder:
    """TT-NN implementation of SparseBox3DEncoder.

    Encodes anchor [bs, N, 11] → embedding [bs, N, embed_dims].
    """

    def __init__(
        self,
        device,
        parameters: dict,
        embed_dims=(128, 32, 32, 64),
        vel_dims: int = 3,
        mode: str = "cat",
        has_output_fc: bool = False,
        in_loops: int = 1,
        out_loops: int = 4,
        use_host_compute: bool = True,
        mesh_device=None,
    ) -> None:
        self.device = mesh_device if mesh_device is not None else device
        self._mesh_device = mesh_device
        self.vel_dims = vel_dims
        self.mode = mode
        self._use_host_compute = use_host_compute
        self._hifi_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False, packer_l1_acc=False, math_approx_mode=False,
        )
        self.has_output_fc = has_output_fc

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.embed_dims = embed_dims
        # Total output dim depends on mode
        if mode == "cat":
            self.output_dims = sum(embed_dims[:3])
            if vel_dims > 0:
                self.output_dims += embed_dims[3]
        else:
            self.output_dims = embed_dims[0]

        # Store host-side parameters for host compute path
        if use_host_compute:
            self.pos_layers_host = self._store_host_params(parameters["pos_fc"])
            self.size_layers_host = self._store_host_params(parameters["size_fc"])
            self.yaw_layers_host = self._store_host_params(parameters["yaw_fc"])
            if vel_dims > 0:
                self.vel_layers_host = self._store_host_params(parameters["vel_fc"])
            if has_output_fc:
                self.out_layers_host = self._store_host_params(parameters["output_fc"])

        # Each embedding_layer has (in_loops * out_loops) Linear+LN pairs
        # Structure: for each out_loop: for each in_loop: Linear→ReLU, then LN
        self.pos_layers = self._load_layers(parameters["pos_fc"], out_loops, in_loops)
        self.size_layers = self._load_layers(parameters["size_fc"], out_loops, in_loops)
        self.yaw_layers = self._load_layers(parameters["yaw_fc"], out_loops, in_loops)
        if vel_dims > 0:
            self.vel_layers = self._load_layers(parameters["vel_fc"], out_loops, in_loops)
        if has_output_fc:
            self.out_layers = self._load_layers(parameters["output_fc"], out_loops, in_loops)

    def _store_host_params(self, layer_params: list) -> list:
        """Store parameters as PyTorch tensors for host computation."""
        stored = []
        for p in layer_params:
            entry = {}
            # weight is already transposed [in, out] for ttnn; transpose back for torch
            entry["weight"] = p["weight"].t().float().clone()  # [out, in] for F.linear
            entry["bias"] = p["bias"].float().clone()
            if "ln_weight" in p:
                entry["ln_weight"] = p["ln_weight"].float().clone()
                entry["ln_bias"] = p["ln_bias"].float().clone()
            stored.append(entry)
        return stored

    def _run_layers_host(self, x: torch.Tensor, layers: list) -> torch.Tensor:
        """Run Linear→ReLU→LN chain on host (PyTorch, float32)."""
        import torch.nn.functional as F
        for entry in layers:
            x = F.linear(x, entry["weight"], entry["bias"])
            x = F.relu(x)
            if "ln_weight" in entry:
                x = F.layer_norm(x, [x.shape[-1]], entry["ln_weight"], entry["ln_bias"], eps=1e-5)
        return x

    def _load_layers(self, layer_params: list, out_loops: int, in_loops: int):
        """Load Linear→ReLU→LN chain parameters.

        layer_params is a list of dicts, each with keys:
          - "weight": Linear weight (already transposed for ttnn)
          - "bias": Linear bias
          - "ln_weight": LayerNorm weight (optional, present after each out_loop)
          - "ln_bias": LayerNorm bias (optional)
        """
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

    def _run_layers(self, x: ttnn.Tensor, layers: list, name: str = "") -> ttnn.Tensor:
        """Run Linear+ReLU (→LN) chain with fused activation."""
        for idx, entry in enumerate(layers):
            linear_in = x
            x = ttnn.linear(x, entry["weight"], bias=entry["bias"],
                            activation="relu",
                            compute_kernel_config=self._hifi_compute_config)
            if idx > 0:
                ttnn.deallocate(linear_in)
            if "ln_weight" in entry:
                ln_in = x
                x = ttnn.layer_norm(
                    x, weight=entry["ln_weight"], bias=entry["ln_bias"],
                    epsilon=1e-5,
                    compute_kernel_config=self._hifi_compute_config,
                )
                ttnn.deallocate(ln_in)
        return x

    def run(
        self,
        box_3d: ttnn.Tensor,
        bs: int,
        num_anchor: int,
    ) -> ttnn.Tensor:
        """Encode anchor boxes to embeddings.

        Args:
            box_3d: [bs, num_anchor, 11] on device (TILE)
            bs: batch size
            num_anchor: number of anchors

        Returns:
            output: [bs, num_anchor, output_dims] on device (TILE)
        """
        # Extract components via device slice (no host roundtrip)
        n = bs * num_anchor

        pos = ttnn.slice(box_3d, [0, 0, X], [bs, num_anchor, Z + 1])        # [bs, 900, 3]
        size = ttnn.slice(box_3d, [0, 0, W], [bs, num_anchor, H + 1])       # [bs, 900, 3]
        yaw = ttnn.slice(box_3d, [0, 0, SIN_YAW], [bs, num_anchor, COS_YAW + 1])  # [bs, 900, 2]

        pos = ttnn.reshape(pos, (1, 1, n, 3))
        size = ttnn.reshape(size, (1, 1, n, 3))
        yaw = ttnn.reshape(yaw, (1, 1, n, 2))

        pos_feat = self._run_layers(pos, self.pos_layers, name="pos")
        ttnn.deallocate(pos)

        size_feat = self._run_layers(size, self.size_layers, name="size")
        ttnn.deallocate(size)

        yaw_feat = self._run_layers(yaw, self.yaw_layers, name="yaw")
        ttnn.deallocate(yaw)

        if self.mode == "add":
            output = ttnn.add(ttnn.add(pos_feat, size_feat), yaw_feat)
        else:
            output = ttnn.concat([pos_feat, size_feat, yaw_feat], dim=-1)
        ttnn.deallocate(pos_feat)
        ttnn.deallocate(size_feat)
        ttnn.deallocate(yaw_feat)

        if self.vel_dims > 0:
            vel = ttnn.slice(box_3d, [0, 0, VX], [bs, num_anchor, VX + self.vel_dims])
            vel = ttnn.reshape(vel, (1, 1, n, self.vel_dims))
            vel_feat = self._run_layers(vel, self.vel_layers, name="vel")
            ttnn.deallocate(vel)
            old_output = output
            if self.mode == "add":
                output = ttnn.add(old_output, vel_feat)
            else:
                output = ttnn.concat([old_output, vel_feat], dim=-1)
            ttnn.deallocate(old_output)
            ttnn.deallocate(vel_feat)

        if self.has_output_fc:
            output = self._run_layers(output, self.out_layers)

        output = ttnn.reshape(output, (bs, num_anchor, self.output_dims))
        return output

    def _run_host(
        self,
        box_pt: torch.Tensor,
        bs: int,
        num_anchor: int,
        n: int,
    ) -> ttnn.Tensor:
        """Run encoder entirely on host (PyTorch float32) for maximum precision.

        The encoder processes small tensors ([bs, 900, 11] → [bs, 900, 256]),
        so host compute has negligible PCIe overhead compared to accuracy gain.
        """
        pos_in = box_pt[:, :, X:Z+1].contiguous().reshape(n, 3)
        size_in = box_pt[:, :, W:H+1].contiguous().reshape(n, 3)
        yaw_in = box_pt[:, :, SIN_YAW:COS_YAW+1].contiguous().reshape(n, 2)

        with torch.no_grad():
            pos_feat = self._run_layers_host(pos_in, self.pos_layers_host)
            size_feat = self._run_layers_host(size_in, self.size_layers_host)
            yaw_feat = self._run_layers_host(yaw_in, self.yaw_layers_host)

            if self.mode == "add":
                output = pos_feat + size_feat + yaw_feat
            else:
                output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

            if self.vel_dims > 0:
                vel_in = box_pt[:, :, VX:VX+self.vel_dims].contiguous().reshape(n, self.vel_dims)
                vel_feat = self._run_layers_host(vel_in, self.vel_layers_host)
                if self.mode == "add":
                    output = output + vel_feat
                else:
                    output = torch.cat([output, vel_feat], dim=-1)

            if self.has_output_fc:
                output = self._run_layers_host(output, self.out_layers_host)

        output = output.reshape(bs, num_anchor, self.output_dims)

        kwargs = dict(layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        if self._mesh_device is not None:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self._mesh_device)
        return ttnn.from_torch(output, **kwargs)


def _extract_linear_relu_ln_params(sequential_module):
    """Extract params from nn.Sequential of Linear→ReLU→LN chains.

    Structure from linear_relu_ln():
      for out_loop:
        for in_loop:
          Linear(in, out)
          ReLU
        LayerNorm(out)

    Returns list of dicts with weight, bias, and optional ln_weight/ln_bias.
    """
    params = []
    if hasattr(sequential_module, 'children'):
        modules = list(sequential_module.children())
    else:
        modules = list(sequential_module)
    i = 0
    while i < len(modules):
        m = modules[i]
        if hasattr(m, 'weight') and hasattr(m, 'in_features'):
            # This is a Linear layer
            entry = {
                "weight": m.weight.data.clone().t(),
                "bias": m.bias.data.clone(),
            }
            # Check if there's a LayerNorm after ReLU (skip ReLU at i+1)
            # Pattern: Linear, ReLU, [LayerNorm | Linear]
            if i + 2 < len(modules) and isinstance(modules[i + 2], torch.nn.LayerNorm):
                ln = modules[i + 2]
                entry["ln_weight"] = ln.weight.data.clone()
                entry["ln_bias"] = ln.bias.data.clone()
                i += 3  # skip Linear, ReLU, LN
            else:
                i += 2  # skip Linear, ReLU
            params.append(entry)
        else:
            i += 1
    return params


def preprocess_encoder_parameters(pt_encoder) -> dict:
    """Extract parameters from SparseBox3DEncoder.

    Args:
        pt_encoder: PyTorch SparseBox3DEncoder instance

    Returns:
        dict with pos_fc, size_fc, yaw_fc, vel_fc, output_fc layer params
    """
    params = {}
    params["pos_fc"] = _extract_linear_relu_ln_params(pt_encoder.pos_fc)
    params["size_fc"] = _extract_linear_relu_ln_params(pt_encoder.size_fc)
    params["yaw_fc"] = _extract_linear_relu_ln_params(pt_encoder.yaw_fc)

    if hasattr(pt_encoder, 'vel_fc'):
        params["vel_fc"] = _extract_linear_relu_ln_params(pt_encoder.vel_fc)

    if pt_encoder.output_fc is not None:
        params["output_fc"] = _extract_linear_relu_ln_params(pt_encoder.output_fc)

    return params
