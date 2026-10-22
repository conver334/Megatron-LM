# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepGEMM bf16 x fp32-split GEMM: the batch-invariant router-gate contract shared with SGLang."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

E, H = 128, 6144


def _dg():
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module

    try:
        return deep_gemm_module()
    except ImportError as exc:
        pytest.skip(str(exc))


def test_split_gemm_matches_fp64_reference_of_the_split_weight():
    from megatron.lite.primitive.kernels.sgl_deepgemm import bf16xfp32_gemm, effective_weight

    _dg()
    torch.manual_seed(0)
    w = torch.randn(E, H, device="cuda") * 0.02
    x = torch.randn(512, H, device="cuda", dtype=torch.bfloat16)
    out = bf16xfp32_gemm(x, w)
    ref = x.double() @ effective_weight(w).double().t()
    torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-4)  # fp32 accumulation of ~6k terms
    exact = x.double() @ w.double().t()
    print(f"\nsplit weight vs exact fp32 weight: max |d| {(ref - exact).abs().max().item():.2e} "
          f"(relative to max |logit| {exact.abs().max().item():.2e}); vs cuBLAS fp32 torch.mm: "
          f"{(out != torch.mm(x.float(), w.t())).float().mean().item():.3f} of logits differ")


def test_split_gemm_is_batch_invariant_and_reproducible():
    from megatron.lite.primitive.kernels.sgl_deepgemm import bf16xfp32_gemm

    _dg()
    torch.manual_seed(1)
    w = torch.randn(E, H, device="cuda") * 0.02
    x = torch.randn(4096, H, device="cuda", dtype=torch.bfloat16)
    full = bf16xfp32_gemm(x, w)
    assert torch.equal(full, bf16xfp32_gemm(x, w))
    report = {n: (bf16xfp32_gemm(x[:n], w) != full[:n]).sum().item() for n in (1, 7, 8, 37, 512, 2048)}
    print(f"\nDeepGEMM split gate: logits mismatches vs the 4096-row launch {report}")
    assert all(v == 0 for v in report.values()), report
    tail = bf16xfp32_gemm(x[4000:], w)
    assert torch.equal(tail, full[4000:])
