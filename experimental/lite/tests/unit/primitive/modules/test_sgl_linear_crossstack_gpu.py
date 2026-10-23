# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cross-stack: SGLang's batch-invariant ``aten::mm`` replacement (fork batch_invariant_ops.py loaded by path)
vs mlite ``linear_bf16`` on model-shaped GEMMs, byte for byte."""

from __future__ import annotations

import functools

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

H = 6144
SHAPES = {"o_proj": (64 * 128, H), "qkv": (H, (64 + 2 * 4) * 128), "gate_up": (H, 2 * 3072), "down": (3072, H), "lm_head": (H, 200064)}
ROWS = (1, 8, 37, 512, 4096)


@functools.lru_cache(maxsize=None)
def _fork_bi():
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module
    from sglang_fork import install_stubs, load

    try:
        deep_gemm_module()  # also turns on deep_gemm deterministic algorithms (what the fork's BI mode now does)
    except ImportError as exc:
        pytest.skip(str(exc))
    install_stubs()
    bi = load("srt/batch_invariant_ops/batch_invariant_ops.py", "sglang.srt.batch_invariant_ops.batch_invariant_ops")
    assert bi._ENABLE_MM_DEEPGEMM
    return bi


@pytest.mark.parametrize("name", list(SHAPES))
def test_fork_mm_override_equals_mlite_linear_bf16(name):
    from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16

    bi = _fork_bi()
    k, n = SHAPES[name]
    torch.manual_seed(3)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)  # [out, in] as both stacks store it
    x_all = torch.randn(max(ROWS), k, device="cuda", dtype=torch.bfloat16)
    full = linear_bf16(x_all, w)
    for m in ROWS:
        x = x_all[:m]
        fork = bi.mm_batch_invariant(x, w.t())  # F.linear(x, w) -> aten::mm(x, w.t()) under --enable-deterministic-inference
        ours = linear_bf16(x, w)
        assert fork.dtype == torch.bfloat16 and torch.equal(fork, ours), (name, m, (fork != ours).sum().item())
        assert torch.equal(ours, full[:m]), (name, m)  # batch-invariant
    print(f"\n{name} [{k}->{n}]: fork mm override == mlite linear_bf16 and M-invariant for M in {ROWS}", flush=True)
