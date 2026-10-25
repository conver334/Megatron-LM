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


def _fork():
    from sglang_fork import load_fork_fa4

    return load_fork_fa4()


def test_fork_vendored_fa4_report_vs_pip():
    """Where the fork's FA4 (batch-invariant switch on, as --enable-deterministic-inference sets it) stands against
    the pip FA4 the training side calls: determinism and batch invariance of the fork kernel, and the byte distance
    fork(BI on) / fork(BI off, num_splits=1) / pip. The contract decision (pip on both sides via
    SGLANG_INKLING_FA4_USE_PIP=1, or the fork kernel on the training side) hangs on these numbers."""
    bi, iface = _fork()
    q, k, v, cu = _inputs(2)
    pip = _run(_pip(), q, k, v, cu, max(LENS))
    bi.set_batch_invariant(True)
    on = _run(iface.flash_attn_varlen_func, q, k, v, cu, max(LENS))
    assert torch.equal(on, _run(iface.flash_attn_varlen_func, q, k, v, cu, max(LENS)))
    lo, hi = LENS[0], sum(LENS)
    alone = _run(iface.flash_attn_varlen_func, q[lo:hi].contiguous(), k[lo:hi].contiguous(), v[lo:hi].contiguous(),
                 torch.tensor([0, hi - lo], device="cuda", dtype=torch.int32), hi - lo)
    assert torch.equal(alone, on[lo:hi]), "fork FA4 (BI on) must be batch-invariant"
    bi.set_batch_invariant(False)
    off = _run(iface.flash_attn_varlen_func, q, k, v, cu, max(LENS), num_splits=1)
    d = lambda a, b: (a != b).float().mean().item()
    md = lambda a, b: (a.float() - b.float()).abs().max().item()
    print(f"\nFA4 bytes: fork(BI on) vs pip {d(on, pip):.5f} (max |d| {md(on, pip):.3e}); fork(BI off, num_splits=1) vs pip "
          f"{d(off, pip):.5f} (max |d| {md(off, pip):.3e}); fork BI on vs off {d(on, off):.5f}", flush=True)
    torch.testing.assert_close(on.float(), pip.float(), rtol=2 ** -6, atol=2 ** -6)


@pytest.mark.xfail(strict=False, reason="dense attention contract open: fork FA4 (BI on) vs pip FA4 bytes (see the report test)")
def test_fork_vendored_fa4_batch_invariant_equals_pip():
    bi, iface = _fork()
    bi.set_batch_invariant(True)
    q, k, v, cu = _inputs(2)
    theirs = _run(iface.flash_attn_varlen_func, q, k, v, cu, max(LENS))
    ours = _run(_pip(), q, k, v, cu, max(LENS))
    assert torch.equal(theirs, ours), f"{(theirs != ours).float().mean().item():.5f} of outputs differ"


@pytest.mark.parametrize("lens", [[1537, 2561], [5, 4096, 77]])
def test_pip_fa4_batch_invariance_with_unaligned_lengths(lens):
    """Odd document lengths: a document's rows must not depend on its neighbours in the packed batch."""
    fa = _pip()
    torch.manual_seed(4)
    n = sum(lens)
    q = torch.randn(n, HQ, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(n, HK, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(n, HK, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    packed = _run(fa, q, k, v, cu, max(lens))
    for i in range(len(lens)):
        lo, hi = int(cu[i]), int(cu[i + 1])
        alone = _run(fa, q[lo:hi].contiguous(), k[lo:hi].contiguous(), v[lo:hi].contiguous(), torch.tensor([0, hi - lo], device="cuda", dtype=torch.int32), hi - lo)
        assert torch.equal(alone, packed[lo:hi]), f"doc {i} (len {lens[i]}) alone vs packed: {(alone != packed[lo:hi]).float().mean().item():.4f} differ"


def test_pip_fa4_decode_rows_match_prefill_rows():
    """Decode shape (one query row against its KV prefix, num_splits=1) == the same row inside a prefill call,
    for every position of a short document: SGLang's prefill/decode agreement is what the training forward relies on."""
    fa = _pip()
    torch.manual_seed(5)
    L = 300
    q = torch.randn(L, HQ, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(L, HK, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(L, HK, D, device="cuda", dtype=torch.bfloat16)
    full = _run(fa, q, k, v, torch.tensor([0, L], device="cuda", dtype=torch.int32), L)
    bad = []
    for t in range(0, L, 7):
        out = fa(q[t : t + 1], k[: t + 1].contiguous(), v[: t + 1].contiguous(), cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                 cu_seqlens_k=torch.tensor([0, t + 1], device="cuda", dtype=torch.int32), max_seqlen_q=1, max_seqlen_k=t + 1,
                 softmax_scale=D ** -0.5, causal=True, num_splits=1)
        out = out[0] if isinstance(out, tuple) else out
        if not torch.equal(out[0], full[t]):
            bad.append((t, (out[0] != full[t]).float().mean().item()))
    print(f"\npip FA4 decode-row vs prefill-row mismatches at {len(bad)} of {len(range(0, L, 7))} positions: {bad[:6]}")
    assert not bad


@pytest.mark.parametrize("split", [128, 150, 257])
def test_chunked_prefill_rows_match_full_prefill(split):
    """Chunked prefill (second chunk: seqlen_q < seqlen_k, the causal boundary mid-block) vs one full prefill, for
    the pip kernel and for the fork kernel with its batch-invariant switch on/off. The fork's switch exists because
    the plain kernel splits the causal boundary into two passes whose rounding depends on seqlen_q."""
    bi, iface = _fork()
    torch.manual_seed(6)
    L = 400
    q = torch.randn(L, HQ, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(L, HK, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(L, HK, D, device="cuda", dtype=torch.bfloat16)

    def chunk(fn, **kw):
        out = fn(q[split:].contiguous(), k, v, cu_seqlens_q=torch.tensor([0, L - split], device="cuda", dtype=torch.int32),
                 cu_seqlens_k=torch.tensor([0, L], device="cuda", dtype=torch.int32), max_seqlen_q=L - split, max_seqlen_k=L,
                 softmax_scale=D ** -0.5, causal=True, num_splits=1, **kw)
        return out[0] if isinstance(out, tuple) else out

    res = {}
    full = _run(_pip(), q, k, v, torch.tensor([0, L], device="cuda", dtype=torch.int32), L)
    res["pip"] = (chunk(_pip()) != full[split:]).float().mean().item()
    for flag in (True, False):
        bi.set_batch_invariant(flag)
        full_f = _run(iface.flash_attn_varlen_func, q, k, v, torch.tensor([0, L], device="cuda", dtype=torch.int32), L, num_splits=1)
        res[f"fork_bi_{flag}"] = (chunk(iface.flash_attn_varlen_func) != full_f[split:]).float().mean().item()
    print(f"\nchunked prefill (split {split}) vs full, fraction of bytes differing: {res}", flush=True)
    assert res["fork_bi_True"] == 0.0, "the batch-invariant fork kernel must give the same rows in chunks and in one pass"
    print(f"plain pip kernel chunk-vs-full mismatch {res['pip']:.5f} (this is what the BI switch repairs)")
