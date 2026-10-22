# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Gemma RMSNorm whose forward is the SGLang inference kernel (sgl_kernel ``gemma_rmsnorm`` /
``gemma_fused_add_rmsnorm``, built from flashinfer ``norm.cu``) and whose backward is the plain fp32 formula.
Forward bytes match inference; backward only has to be a correct gradient.

SGLang calls the post-attention and final norms with a residual: the kernel forms ``a + r`` in fp32, normalises
that fp32 sum and writes the bf16 sum back as the new residual. ``gemma_fused_add_rmsnorm`` below is that
contract; the standalone ``SglGemmaRMSNorm.forward`` matches only norms called without a residual."""

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


def gemma_fused_add_rmsnorm_kernel():
    try:
        from sgl_kernel import gemma_fused_add_rmsnorm

        return gemma_fused_add_rmsnorm
    except ImportError:
        pass
    try:
        from flashinfer.norm import gemma_fused_add_rmsnorm

        return gemma_fused_add_rmsnorm
    except ImportError as exc:
        raise ImportError("align rmsnorm needs sgl_kernel or flashinfer-python (gemma_fused_add_rmsnorm)") from exc


def _rmsnorm_backward(g: torch.Tensor, xf: torch.Tensor, w: torch.Tensor, eps: float):
    """fp32 gradient of y = x * rsqrt(mean(x^2) + eps) * (1 + w) -> (dx, dw)."""
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    xhat = xf * rstd
    gw = (g * xhat).sum(0)
    gxhat = g * (1.0 + w.float())
    gx = rstd * (gxhat - xhat * (gxhat * xhat).mean(-1, keepdim=True))
    return gx, gw


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
        gx, gw = _rmsnorm_backward(grad.contiguous().reshape(x2.shape).float(), x2.float(), w, ctx.eps)
        return gx.to(x2.dtype).reshape(grad.shape), gw.to(ctx.weight_dtype), None


class _GemmaFusedAddRMSNormFn(torch.autograd.Function):
    """(normed, new_residual) = kernel(a, r): new_residual = bf16(a + r), normed = norm(fp32(a + r))."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, r: torch.Tensor, weight: torch.Tensor, eps: float):
        shape = a.shape
        hid = a.contiguous().reshape(-1, shape[-1]).clone()  # kernel: hid <- normed
        res = r.contiguous().reshape(-1, shape[-1]).clone()  # kernel: res <- a + r
        w = weight.to(a.dtype)
        gemma_fused_add_rmsnorm_kernel()(hid, res, w, eps)
        ctx.save_for_backward(a, r, w)
        ctx.eps = eps
        ctx.weight_dtype = weight.dtype
        return hid.reshape(shape), res.reshape(shape)

    @staticmethod
    def backward(ctx, g_out: torch.Tensor, g_res: torch.Tensor | None):
        a, r, w = ctx.saved_tensors
        yf = (a.float() + r.float()).reshape(-1, a.shape[-1])
        gx, gw = _rmsnorm_backward(g_out.contiguous().reshape(yf.shape).float(), yf, w, ctx.eps)
        if g_res is not None:
            gx = gx + g_res.contiguous().reshape(yf.shape).float()
        gx = gx.reshape(a.shape)
        return gx.to(a.dtype), gx.to(r.dtype), gw.to(ctx.weight_dtype), None


def gemma_fused_add_rmsnorm(a: torch.Tensor, r: torch.Tensor, weight: torch.Tensor, eps: float):
    """SGLang ``norm(hidden, residual)``: returns (normed, residual + hidden) with inference bytes."""
    return _GemmaFusedAddRMSNormFn.apply(a, r, weight, eps)


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
