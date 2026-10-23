# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Vendored SGLang BI log_softmax: stats twin == full kernel at the label, == the fork's kernel by path, M-invariant."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

V = 200064


def test_stats_twin_matches_full_kernel_and_torch_within_tolerance():
    from megatron.lite.primitive.kernels.sgl_log_softmax import log_softmax, token_logprob

    torch.manual_seed(0)
    x_bf16 = torch.randn(64, V, device="cuda", dtype=torch.bfloat16) * 4
    x = x_bf16.float()  # SGLang: logits.float() then the BI kernel
    labels = torch.randint(0, V, (64,), device="cuda")
    full = torch.gather(log_softmax(x), 1, labels[:, None]).squeeze(1)
    ours = token_logprob(x, labels)
    assert torch.equal(ours, full)
    assert torch.equal(token_logprob(x_bf16, labels), full)  # bf16 logits converted per row chunk like SGLang's .float()
    ref = torch.gather(torch.log_softmax(x, -1), 1, labels[:, None]).squeeze(1)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)
    print(f"\nBI log_softmax vs torch.log_softmax at the label: {(ours != ref).sum().item()}/{ours.numel()} fp32 mismatches")
    assert torch.equal(token_logprob(x[:7], labels[:7]), ours[:7])  # per-row kernel: batch-invariant
    from megatron.lite.primitive.kernels.sgl_log_softmax import log_softmax_stats

    mx, lse = log_softmax_stats(x_bf16, chunk_rows=5)  # chunking must not change bytes
    mx2, lse2 = log_softmax_stats(x)
    assert torch.equal(mx, mx2) and torch.equal(lse, lse2)


def test_vendored_kernel_equals_fork_kernel_by_path():
    from megatron.lite.primitive.kernels.sgl_log_softmax import log_softmax
    from sglang_fork import install_stubs, load

    install_stubs()
    bi = load("srt/batch_invariant_ops/batch_invariant_ops.py", "sglang.srt.batch_invariant_ops.batch_invariant_ops")
    torch.manual_seed(1)
    x = (torch.randn(16, V, device="cuda", dtype=torch.bfloat16) * 4).float()
    assert torch.equal(log_softmax(x), bi.log_softmax(x, dim=-1))
