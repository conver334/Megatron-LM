# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense attention contract (L0-2): the pip FA4 forward the training side calls (flash_attn.cute
flash_attn_varlen_func, causal, num_splits=1) is deterministic, batch-invariant and layout-invariant (paged KV ==
contiguous), and equals the fork's vendored FA4 run by path with its batch-invariant switch on."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

HQ, HK, D = 64, 4, 128
LENS = [1536, 2560]


def _inputs(seed=0):
    torch.manual_seed(seed)
    n = sum(LENS)
    q = torch.randn(n, HQ, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(n, HK, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(n, HK, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, LENS[0], n], device="cuda", dtype=torch.int32)
    return q, k, v, cu


def _pip():
    from flash_attn.cute.interface import flash_attn_varlen_func

    return flash_attn_varlen_func


def _run(fn, q, k, v, cu, max_len, **kw):
    out = fn(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_len, max_seqlen_k=max_len, softmax_scale=D ** -0.5, causal=True, **kw)
    return out[0] if isinstance(out, tuple) else out


def test_pip_fa4_varlen_is_deterministic_batch_invariant_and_split_invariant():
    fa = _pip()
    q, k, v, cu = _inputs()
    packed = _run(fa, q, k, v, cu, max(LENS))
    assert torch.equal(packed, _run(fa, q, k, v, cu, max(LENS)))
    assert torch.equal(packed, _run(fa, q, k, v, cu, max(LENS), num_splits=1))
    for i, (lo, hi) in enumerate(zip([0, LENS[0]], [LENS[0], sum(LENS)])):
        alone = _run(fa, q[lo:hi].contiguous(), k[lo:hi].contiguous(), v[lo:hi].contiguous(),
                     torch.tensor([0, hi - lo], device="cuda", dtype=torch.int32), hi - lo)
        assert torch.equal(alone, packed[lo:hi]), f"doc {i} alone vs packed"


def test_pip_fa4_paged_kv_equals_contiguous():
    """SGLang serves K/V from a paged pool (page_table, page 128); the training side packs them contiguously."""
    fa = _pip()
    q, k, v, cu = _inputs(1)
    packed = _run(fa, q, k, v, cu, max(LENS))
    page = 128
    npages = sum(LENS) // page
    perm = torch.randperm(npages, generator=torch.Generator(device="cuda").manual_seed(0), device="cuda")
    kc = torch.empty(npages, page, HK, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.empty_like(kc)
    kc[perm] = k.view(npages, page, HK, D)
    vc[perm] = v.view(npages, page, HK, D)
    pt = torch.full((2, max(LENS) // page), -1, device="cuda", dtype=torch.int32)
    pt[0, : LENS[0] // page] = perm[: LENS[0] // page]
    pt[1, : LENS[1] // page] = perm[LENS[0] // page :]
    out = fa(q, kc, vc, cu_seqlens_q=cu, max_seqlen_q=max(LENS), seqused_k=torch.tensor(LENS, device="cuda", dtype=torch.int32),
             page_table=pt, softmax_scale=D ** -0.5, causal=True)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.equal(out, packed)


def test_fork_vendored_fa4_batch_invariant_equals_pip():
    """The fork's FA4 (set_batch_invariant(True), what --enable-deterministic-inference turns on) == pip FA4 bytes."""
    from sglang_fork import load_fork_fa4

    bi, iface = load_fork_fa4()
    bi.set_batch_invariant(True)
    q, k, v, cu = _inputs(2)
    theirs = _run(iface.flash_attn_varlen_func, q, k, v, cu, max(LENS))
    ours = _run(_pip(), q, k, v, cu, max(LENS))
    assert torch.equal(theirs, ours), f"{(theirs != ours).float().mean().item():.5f} of outputs differ"
    assert torch.equal(theirs, _run(iface.flash_attn_varlen_func, q, k, v, cu, max(LENS)))
    bi.set_batch_invariant(False)
    off = _run(iface.flash_attn_varlen_func, q, k, v, cu, max(LENS), num_splits=1)
    print(f"\nfork FA4 batch-invariant off (num_splits=1) vs on: {(off != theirs).float().mean().item():.5f} of outputs differ")
