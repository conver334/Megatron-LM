# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Gemma RMSNorm whose forward is the SGLang inference kernel (sgl_kernel ``gemma_rmsnorm``, built from
flashinfer ``norm.cu``) and whose backward is the plain fp32 formula. Forward bytes match inference;
backward only has to be a correct gradient."""

from __future__ import annotations

import torch
import torch.nn as nn


def gemma_rmsnorm_kernel():
    """sgl_kernel when importable, else pip flashinfer (same ``norm.cu`` source, JIT against local torch)."""
    try:
        from sgl_kernel import gemma_rmsnorm

        return gemma_rmsnorm
    except ImportError:
        pass
    try:
        from flashinfer.norm import gemma_rmsnorm

        return gemma_rmsnorm
    except ImportError as exc:
        raise ImportError("align rmsnorm needs sgl_kernel or flashinfer-python (gemma_rmsnorm)") from exc


class _GemmaRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x2 = x.contiguous().reshape(-1, x.shape[-1])
        w = weight.to(x2.dtype)  # inference holds the norm weight in the activation dtype
        out = gemma_rmsnorm_kernel()(x2, w, eps)
        ctx.save_for_backward(x2, w)
        ctx.eps = eps
        ctx.weight_dtype = weight.dtype
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x2, w = ctx.saved_tensors
        xf = x2.float()
        g = grad.contiguous().reshape(x2.shape).float()
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + ctx.eps)
        xhat = xf * rstd
        gw = (g * xhat).sum(0)
        gxhat = g * (1.0 + w.float())
        gx = rstd * (gxhat - xhat * (gxhat * xhat).mean(-1, keepdim=True))
        return gx.to(x2.dtype).reshape(grad.shape), gw.to(ctx.weight_dtype), None


class SglGemmaRMSNorm(nn.Module):
    """Drop-in for ``te.RMSNorm(zero_centered_gamma=True)``: same ``weight`` name and (1 + w) semantics."""

    def __init__(self, hidden_size: int, eps: float, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))

    @classmethod
    def from_te(cls, module: nn.Module, eps: float) -> "SglGemmaRMSNorm":
        new = cls(module.weight.numel(), eps, device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(module.weight)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GemmaRMSNormFn.apply(x, self.weight, self.eps)
