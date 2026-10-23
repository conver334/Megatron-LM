# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Triton swigluoai (contract activation) vs fp32 reference, SGLang's torch.compile formula and mlite's quick_geglu."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

ALPHA, LIMIT = 1.702, 7.0


def _ref(x, alpha, limit):
    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(-limit, limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)


@torch.compile
def _sglang_compiled(x, gemm1_alpha, gemm1_limit):  # verbatim: sglang fused_moe.swiglu_no_interleaved_with_alpha_and_limit
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


@pytest.mark.parametrize("shape", [(64, 2 * 3072), (3, 5, 2 * 1000)])
def test_swigluoai_matches_fp32_reference_and_handles_tails(shape):
    from megatron.lite.primitive.kernels.sgl_swiglu import swigluoai

    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 4  # exercises both clamps
    out = swigluoai(x, ALPHA, LIMIT)
    assert out.shape == (*shape[:-1], shape[-1] // 2) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), _ref(x, ALPHA, LIMIT), rtol=1e-2, atol=1e-2)
    assert torch.equal(out, swigluoai(x, ALPHA, LIMIT))


def test_swigluoai_vs_sglang_compiled_and_mlite_quick_geglu():
    from types import SimpleNamespace

    from megatron.lite.model.minimax_m3.lite.model import m3_activation
    from megatron.lite.primitive.kernels.sgl_swiglu import swigluoai

    torch.manual_seed(1)
    x = torch.randn(4096, 2 * 3072, device="cuda", dtype=torch.bfloat16) * 4
    ours = swigluoai(x, ALPHA, LIMIT)
    compiled = _sglang_compiled(x, ALPHA, LIMIT)
    mlite = m3_activation(x, SimpleNamespace(swiglu_alpha=ALPHA, swiglu_limit=LIMIT, swiglu_up_offset=1.0))
    torch.testing.assert_close(ours.float(), compiled.float(), rtol=1e-2, atol=1e-2)
    print(f"\nswigluoai Triton vs SGLang torch.compile (this torch): {(ours != compiled).sum().item()}/{ours.numel()} bf16 mismatches; "
          f"vs mlite quick_geglu: {(ours != mlite).sum().item()}/{ours.numel()}", flush=True)
