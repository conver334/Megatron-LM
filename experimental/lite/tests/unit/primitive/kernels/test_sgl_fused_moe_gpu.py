# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Vendored SGLang fused-MoE path: kernel text == fork, kernel launch == fork's launcher by path, torch
re-implementations of moe_align_block_size / moe_sum_reduce behave as the CUDA ops, fp64 sanity."""

from __future__ import annotations

import inspect
import re

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

T, H, I, E, K = 96, 512, 256, 16, 4
ALPHA, LIMIT = 1.702, 7.0


def _inputs(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    w13 = (torch.randn(E, 2 * I, H, device="cuda") * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda") * 0.02).to(torch.bfloat16)
    logits = torch.randn(T, E, device="cuda")
    bias = torch.randn(E, device="cuda") * 0.05
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate

    weights, ids = moe_fused_gate(logits, bias, K, routed_scaling_factor=2.0, apply_routed_scaling_factor_on_output=True)
    return x, w13, w2, weights, ids


def test_moe_align_block_size_layout():
    from megatron.lite.primitive.kernels.sgl_fused_moe import moe_align_block_size

    topk_ids = torch.tensor([[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], device="cuda", dtype=torch.int32) - 1  # docstring example (0-based)
    sorted_ids, expert_ids, n_post = moe_align_block_size(topk_ids, 4, 4)
    assert n_post.item() == 16 and sorted_ids.shape[0] == 16 and expert_ids.tolist() == [0, 1, 2, 3]
    numel = topk_ids.numel()
    for e in range(4):
        blk = sorted_ids[4 * e : 4 * e + 4].tolist()
        real = [i for i in blk if i != numel]
        assert all(topk_ids.reshape(-1)[i].item() == e for i in real) and len(real) == 3
    assert sorted(i for i in sorted_ids.tolist() if i != numel) == list(range(numel))


def test_fused_experts_matches_fp64_semantics_and_is_deterministic():
    from megatron.lite.primitive.kernels.sgl_fused_moe import fused_experts_bf16

    x, w13, w2, weights, ids = _inputs()
    out = fused_experts_bf16(x, w13, w2, weights, ids, ALPHA, LIMIT)
    assert torch.equal(out, fused_experts_bf16(x, w13, w2, weights, ids, ALPHA, LIMIT))
    ref = torch.zeros(T, H, device="cuda", dtype=torch.float64)
    for k in range(K):
        e = ids[:, k].long()
        gu = torch.einsum("th,tnh->tn", x.double(), w13.double()[e])
        gate, up = gu.chunk(2, dim=-1)
        act = gate.clamp(max=LIMIT) * torch.sigmoid(gate.clamp(max=LIMIT) * ALPHA) * (up.clamp(-LIMIT, LIMIT) + 1)
        ref += torch.einsum("ti,thi->th", act, w2.double()[e]) * weights[:, k].double()[:, None]
    torch.testing.assert_close(out.double(), ref, rtol=3e-2, atol=3e-2)


def test_vendored_kernel_text_equals_fork_and_launch_equals_fork_launcher():
    from megatron.lite.primitive.kernels import sgl_fused_moe as V
    from megatron.lite.primitive.kernels.sgl_fused_moe import invoke_fused_moe_kernel_bf16, moe_align_block_size
    from sglang_fork import fork_root, install_stubs, load

    install_stubs()
    fork = load("kernels/ops/moe/fused_moe_triton_kernels.py", "sglang.kernels.ops.moe.fused_moe_triton_kernels")
    for name in ("fused_moe_kernel", "write_zeros_to_output"):
        ours = inspect.getsource(getattr(V, name).fn) if hasattr(getattr(V, name), "fn") else inspect.getsource(getattr(V, name))
        theirs = inspect.getsource(getattr(fork, name).fn) if hasattr(getattr(fork, name), "fn") else inspect.getsource(getattr(fork, name))
        assert re.sub(r"\s+", " ", ours) == re.sub(r"\s+", " ", theirs), name
    x, w13, w2, weights, ids = _inputs(1)
    import triton.language as tl

    sorted_ids, expert_ids, n_post = moe_align_block_size(ids, 64, E)
    ours = torch.empty(T * K, 2 * I, dtype=torch.bfloat16, device="cuda")
    theirs = torch.empty_like(ours)
    invoke_fused_moe_kernel_bf16(x, w13, ours, weights, ids, sorted_ids, expert_ids, n_post, False, K)
    fork.invoke_fused_moe_kernel(
        x, w13, None, theirs, None, None, None, weights, ids, sorted_ids, expert_ids, n_post, False, K,
        dict(V.DETERMINISTIC_CONFIG), compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False, block_shape=None, filter_expert=False, router_topk=K,
    )
    assert torch.equal(ours, theirs)
    act = V.swigluoai(ours, ALPHA, LIMIT)
    ours3 = torch.empty(T, K, H, dtype=torch.bfloat16, device="cuda")
    theirs3 = torch.empty_like(ours3)
    invoke_fused_moe_kernel_bf16(act, w2, ours3, weights, ids, sorted_ids, expert_ids, n_post, True, 1)
    fork.invoke_fused_moe_kernel(
        act, w2, None, theirs3, None, None, None, weights, ids, sorted_ids, expert_ids, n_post, True, 1,
        dict(V.DETERMINISTIC_CONFIG), compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False, block_shape=None, filter_expert=False, router_topk=K,
    )
    assert torch.equal(ours3, theirs3)


def test_moe_sum_reduce_is_sequential_fp32():
    from megatron.lite.primitive.kernels.sgl_fused_moe import moe_sum_reduce

    torch.manual_seed(3)
    c3 = torch.randn(T, K, H, device="cuda", dtype=torch.bfloat16)
    out = moe_sum_reduce(c3)
    seq = ((c3[:, 0].float() + c3[:, 1].float()) + c3[:, 2].float()) + c3[:, 3].float()
    assert torch.equal(out, seq.to(torch.bfloat16))
    print(f"\nmoe_sum_reduce vs torch.sum(dim=1): {(out != c3.float().sum(1).to(torch.bfloat16)).sum().item()}/{out.numel()} bf16 mismatches")
