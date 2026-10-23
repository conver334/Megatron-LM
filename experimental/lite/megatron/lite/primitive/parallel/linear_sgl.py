# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TE linear wrappers whose forward shows SGLang's batch-invariant GEMM bytes (DeepGEMM bf16, see
``kernels.sgl_deepgemm.linear_bf16``) while the backward stays TE's. Applied by instance class swap so the
``linear.weight`` parameter and checkpoint names are untouched."""

from __future__ import annotations

import torch

from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
from megatron.lite.primitive.kernels.visible import visible_forward
from megatron.lite.primitive.parallel.linear import ColumnParallelLinear, RowParallelLinear


class SglRowParallelLinear(RowParallelLinear):
    @classmethod
    def swap_in(cls, module: RowParallelLinear) -> "SglRowParallelLinear":
        if module.tp_size != 1 or getattr(module.linear, "use_bias", False):
            raise NotImplementedError("align linear: tp=1, no bias")
        module.__class__ = cls
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native = self.linear(x)
        return visible_forward(linear_bf16(x, self.linear.weight), native)


class SglColumnParallelLinear(ColumnParallelLinear):
    """Fused-norm QKV projection shown as SGLang computes it: ``input_layernorm`` = ``gemma_fused_add_rmsnorm``
    of the previous layer's (delta, residual) pair (fp32 sum; standalone ``gemma_rmsnorm`` when there is no
    pair, i.e. layer 0 / embeddings), then the batch-invariant bf16 GEMM. The TE LayerNormLinear graph of the
    bf16 residual stays the native (backward) path. The layer sets ``_residual_pair`` before calling attention."""

    _residual_pair = None

    @classmethod
    def swap_in(cls, module: ColumnParallelLinear) -> "SglColumnParallelLinear":
        if module.tp_size != 1 or getattr(module.linear, "use_bias", False) or not hasattr(module.linear, "layer_norm_weight"):
            raise NotImplementedError("align qkv: tp=1, no bias, fused RMSNorm (LayerNormLinear) only")
        module.__class__ = cls
        return module

    def _visible_norm(self, x: torch.Tensor) -> torch.Tensor:
        from megatron.lite.primitive.kernels.sgl_norm import gemma_fused_add_rmsnorm_kernel, gemma_rmsnorm_kernel

        w = self.linear.layer_norm_weight.detach().to(x.dtype)
        pair = self._residual_pair
        if pair is None:
            return gemma_rmsnorm_kernel()(x.reshape(-1, x.shape[-1]).contiguous(), w, self.linear.eps).reshape(x.shape)
        delta, residual = pair
        hid = delta.detach().reshape(-1, x.shape[-1]).contiguous().clone()
        res = residual.detach().reshape(-1, x.shape[-1]).contiguous().clone()
        gemma_fused_add_rmsnorm_kernel()(hid, res, w, self.linear.eps)
        return hid.reshape(x.shape)

    def forward(self, x: torch.Tensor):
        native = self.linear(x)
        norm_native = None
        if self.return_layernorm_output:
            native, norm_native = native
        with torch.no_grad():
            normed = self._visible_norm(x)
            qkv = linear_bf16(normed, self.linear.weight)
        out = visible_forward(qkv, native)
        if norm_native is None:
            return out
        return out, visible_forward(normed, norm_native)
