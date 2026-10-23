# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepGEMM bf16 GEMMs as used by SGLang's batch-invariant mode (``sgl-deep-gemm``), forward only.

``bf16xfp32_gemm`` is the train/inference contract for GEMMs with an fp32 weight (router gate): the weight
is split into bf16 high/low parts exactly like SGLang's ``_get_bf16xfp32_weight_split`` (scale 1/256), each
part goes through ``deep_gemm.bf16_gemm_nt`` with an fp32 output, and the two fp32 results are summed as
``hi + lo * (1/256)`` (the power-of-two scaling is exact, so the sum has a single rounding)."""

from __future__ import annotations

import torch

WEIGHT_SPLIT_SCALE = 1.0 / 256.0  # SGLang dsv4/gemm.py _HPC_GEMM_WEIGHT_SCALE


def deep_gemm_module():
    """sgl-deep-gemm with deterministic algorithms on: the default routes small-M shapes to cuBLASLt, which is not
    batch-invariant (GB200 10-22: M=1..512 vs 4096 differ); with the flag every call form is bitwise M-invariant."""
    try:
        import deep_gemm
    except ImportError as exc:
        raise ImportError("align deepgemm GEMMs need the sgl-deep-gemm package (import deep_gemm)") from exc
    if not deep_gemm.get_deterministic_algorithms():
        deep_gemm.use_deterministic_algorithms(True)
    return deep_gemm


def split_fp32_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w ~= w_high + WEIGHT_SPLIT_SCALE * w_low, both bf16 (SGLang split)."""
    w = w.detach().float()
    w_high = w.to(torch.bfloat16)
    w_low = ((w - w_high.float()) / WEIGHT_SPLIT_SCALE).to(torch.bfloat16)
    return w_high, w_low


def bf16_gemm_f32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] bf16 @ w [N, K] bf16 -> [M, N] fp32 via deep_gemm.bf16_gemm_nt (fp32 accumulate and output)."""
    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
    deep_gemm_module().bf16_gemm_nt(x.contiguous(), w.contiguous(), out)
    return out


def linear_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """SGLang batch-invariant ``F.linear`` for bf16 weights: ``aten::mm`` -> ``matmul_persistent`` ->
    ``deep_gemm.bf16_gemm_nn(x, W.t(), out_bf16)``. Same call form here (NN with the transposed-weight view)."""
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty(x2.shape[0], w.shape[0], dtype=x2.dtype, device=x2.device)
    deep_gemm_module().bf16_gemm_nn(x2, w.contiguous().t(), out)
    return out.reshape(*x.shape[:-1], w.shape[0])


def bf16xfp32_gemm(x: torch.Tensor, w_fp32: torch.Tensor) -> torch.Tensor:
    """x bf16 @ w_fp32.T with the bf16 high/low split; identical on the SGLang side (algo ``deep_gemm``)."""
    w_high, w_low = split_fp32_weight(w_fp32)
    hi = bf16_gemm_f32(x, w_high)
    lo = bf16_gemm_f32(x, w_low)
    return hi + lo * WEIGHT_SPLIT_SCALE


def effective_weight(w_fp32: torch.Tensor) -> torch.Tensor:
    """The fp32 operand the split GEMM actually multiplies by (for the backward)."""
    w_high, w_low = split_fp32_weight(w_fp32)
    return w_high.float() + w_low.float() * WEIGHT_SPLIT_SCALE
