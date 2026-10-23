# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""swigluoai activation as one Triton kernel: fp32 math, single bf16 rounding, same expression order as
SGLang's ``swiglu_no_interleaved_with_alpha_and_limit`` (``gate * sigmoid(gate * alpha) * (up + 1)`` with
``gate <= limit``, ``-limit <= up <= limit``). Forward only; the contract kernel both stacks run (SGLang's
torch.compile version is Inductor-codegen and not stable across torch versions)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swigluoai_kernel(x_ptr, out_ptr, stride_x, stride_o, alpha, limit, HALF: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < HALF
    gate = tl.load(x_ptr + row * stride_x + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride_x + HALF + col, mask=mask, other=0.0).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    y = gate * tl.sigmoid(gate * alpha) * (up + 1.0)
    tl.store(out_ptr + row * stride_o + col, y.to(out_ptr.dtype.element_ty), mask=mask)


def swigluoai(x: torch.Tensor, alpha: float, limit: float, block: int = 1024) -> torch.Tensor:
    """x [..., 2*I] (gate | up, not interleaved) -> [..., I], same dtype."""
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    half = x2.shape[1] // 2
    out = torch.empty(x2.shape[0], half, dtype=x2.dtype, device=x2.device)
    if x2.shape[0] > 0:
        grid = (x2.shape[0], triton.cdiv(half, block))
        _swigluoai_kernel[grid](x2, out, x2.stride(0), out.stride(0), float(alpha), float(limit), HALF=half, BLOCK=block)
    return out.reshape(*x.shape[:-1], half)
