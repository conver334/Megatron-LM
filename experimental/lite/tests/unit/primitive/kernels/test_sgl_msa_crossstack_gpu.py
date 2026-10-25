# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax sparse attention, training vs inference on identical inputs: Magi's ``msa_v1`` kernels (indexer GEMM +
CuTe top-k, sparse attention) against the SGLang fork's prefill path run by path (CuTe ``inference.msa_v1`` indexer
GEMM + CUDA radix top-k, gathered and paged sparse attention over the shared ``msa_v1`` kernel). Top-k sets and
attention bytes must agree, including exact score ties, packed documents and long sequences."""

from __future__ import annotations

import importlib

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

HQ, HKV, D, HI, K, BLK = 64, 4, 128, 4, 16, 128
SCALE = D ** -0.5


def _case(lens, seed, tie_blocks=None):
    torch.manual_seed(seed)
    T = sum(lens)
    q = (torch.randn(T, HQ, D, device="cuda") * 0.5).to(torch.bfloat16)
    k = (torch.randn(T, HKV, D, device="cuda") * 0.5).to(torch.bfloat16)
    v = (torch.randn(T, HKV, D, device="cuda") * 0.5).to(torch.bfloat16)
    iq = (torch.randn(T, HI, D, device="cuda") * 0.5).to(torch.bfloat16)
    ik = (torch.randn(T, 1, D, device="cuda") * 0.5).to(torch.bfloat16)
    if tie_blocks:  # identical key blocks -> exactly tied fp32 block scores
        for d in tie_blocks[1:]:
            ik[d * BLK:(d + 1) * BLK] = ik[tie_blocks[0] * BLK:(tie_blocks[0] + 1) * BLK]
    cu = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(), device="cuda", dtype=torch.int32)
    return q, k, v, iq, ik, cu, torch.tensor(lens, device="cuda", dtype=torch.int32)


def _paged(x, lens, cu):
    """Per-document pages of BLK tokens (SGLang's page128 KV layout) and the page table."""
    nblk = [(l + BLK - 1) // BLK for l in lens]
    out = torch.zeros(sum(nblk) * BLK, *x.shape[1:], device="cuda", dtype=x.dtype)
    table = torch.full((len(lens), max(nblk)), -1, device="cuda", dtype=torch.int32)
    pg = 0
    for b, l in enumerate(lens):
        out[pg * BLK: pg * BLK + l] = x[int(cu[b]): int(cu[b + 1])]
        table[b, : nblk[b]] = torch.arange(pg, pg + nblk[b], device="cuda", dtype=torch.int32)
        pg += nblk[b]
    return out, table, sum(nblk)


def _training(q, k, v, iq, ik, cu, lens, total_pages):
    tr_idx, tr_attn = importlib.import_module("msa_v1.indexer"), importlib.import_module("msa_v1.attention")
    T, max_len = int(cu[-1]), int(lens.max())
    topk, _ = tr_idx.forward(iq, ik, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max_len, max_seqlen_kv=max_len, deterministic=True)
    meta = tr_attn.prepare(topk, cu, cu, total_k=T, total_rows=total_pages, max_seqlen_q=max_len, max_seqlen_k=max_len,
                           fragment_indices=None, prepare_kl_schedule=False)
    out, _ = tr_attn.forward(q, k, v, meta, softmax_scale=SCALE, return_softmax_lse=True, partial_dtype=torch.bfloat16)
    return topk, out


def _fork(ci, cw, q, k, v, iq, ik, cu, lens):
    T, max_len = int(cu[-1]), int(lens.max())
    ik_pages, table, total_pages = _paged(ik, lens.tolist(), cu)
    ws = ci.prepare_m3_prefill_indexer(cu, lens, total_q=T, max_seqlen_q=max_len, max_seqlen_k=max_len, topk=K, num_index_heads=HI)
    topk = ci.minimax_m3_cute_prefill_indexer_topk(iq, ik_pages, table, cu, lens, max_seqlen_q=max_len, max_seqlen_k=max_len, topk=K,
                                                    init_blocks=0, local_blocks=1, sm_scale=SCALE, workspace=ws).clone()
    gathered, _, _ = cw.cute_topk_sparse_attention(q, k, v, topk, cu, cu, max_len, max_len, BLK, K, SCALE)
    kc, _, _ = _paged(k, lens.tolist(), cu)
    vc, _, _ = _paged(v, lens.tolist(), cu)
    paged, _, _ = cw.cute_paged_topk_sparse_attention(q, kc, vc, topk, table, lens, cu, cu, max_len, max_len, BLK, K, BLK, SCALE,
                                                      total_seqblocks_k=total_pages)
    return topk, gathered, paged, ws.score, total_pages


def _sets(t):
    return torch.sort(t.to(torch.int64), dim=-1).values


CASES = {
    "single_4096": ([4096], 0, None),
    "long_32768": ([32768], 1, None),
    "packed_1500_2596": ([1500, 2596], 2, None),
    "packed_4_docs_incl_1_token": ([700, 4096, 1, 2048], 3, None),
    "ties_3_way": ([4096], 4, (3, 7, 11)),
    "ties_16_way": ([4096], 5, tuple(range(16))),
    "ties_20_way_long": ([32768], 6, tuple(range(5, 25))),
}


@pytest.mark.parametrize("name", list(CASES))
def test_msa_prefill_training_equals_fork(name):
    from sglang_fork import load_fork_msa

    ci, cw = load_fork_msa()
    lens, seed, ties = CASES[name]
    q, k, v, iq, ik, cu, lens_t = _case(lens, seed, ties)
    fk_topk, gathered, paged, _, total_pages = _fork(ci, cw, q, k, v, iq, ik, cu, lens_t)
    tr_topk, tr_out = _training(q, k, v, iq, ik, cu, lens_t, total_pages)
    assert tr_topk.shape == fk_topk.shape == (HI, int(cu[-1]), K)
    same = (_sets(tr_topk) == _sets(fk_topk)).all(dim=-1)
    assert same.all(), f"top-k sets differ on {(~same).float().mean().item():.4f} of (head, query) rows"
    assert torch.equal(tr_out, gathered), f"training attention vs fork gathered: {(tr_out != gathered).float().mean().item():.6f} differ"
    assert torch.equal(gathered, paged), f"fork gathered vs paged: {(gathered != paged).float().mean().item():.6f} differ"


def test_msa_fork_path_is_deterministic_and_order_only():
    """Two runs agree bitwise; the fork's top-k order (not a set) is what reaches the attention CSR, and the attention
    bytes do not depend on that order."""
    from sglang_fork import load_fork_msa

    ci, cw = load_fork_msa()
    q, k, v, iq, ik, cu, lens = _case([4096, 2048], 9)
    a = _fork(ci, cw, q, k, v, iq, ik, cu, lens)
    b = _fork(ci, cw, q, k, v, iq, ik, cu, lens)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    # fp32 block scores: the kernel writes only the causal (past) blocks of each query, the rest of the buffer is uninitialised
    pos = torch.cat([torch.arange(int(l), device="cuda") for l in lens.tolist()])
    valid = (torch.arange(a[3].shape[-1], device="cuda")[None, :] <= (pos // BLK)[:, None])[None]
    assert torch.equal(torch.where(valid, a[3], 0.0), torch.where(valid, b[3], 0.0))
    max_len = int(lens.max())
    perm = torch.argsort(torch.rand(a[0].shape, device="cuda"), dim=-1)
    shuffled = torch.gather(a[0], -1, perm)
    out_shuffled, _, _ = cw.cute_topk_sparse_attention(q, k, v, shuffled, cu, cu, max_len, max_len, BLK, K, SCALE)
    assert torch.equal(out_shuffled, a[1])
