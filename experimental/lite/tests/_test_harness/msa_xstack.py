# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared glue for the MSA cross-stack checks: SGLang's prefill chain (CuTe indexer + radix top-k + paged sparse
attention) run by path on packed documents given as flat [T, heads, 128] bf16 tensors."""

from __future__ import annotations

import torch

HQ, HKV, D, HI, K, BLK = 64, 4, 128, 4, 16, 128
SCALE = D ** -0.5


def paged(x: torch.Tensor, lens: list[int], cu: torch.Tensor):
    """Per-document pages of BLK tokens (SGLang's page128 KV layout) and the page table."""
    nblk = [(l + BLK - 1) // BLK for l in lens]
    out = torch.zeros(sum(nblk) * BLK, *x.shape[1:], device=x.device, dtype=x.dtype)
    table = torch.full((len(lens), max(nblk)), -1, device=x.device, dtype=torch.int32)
    pg = 0
    for b, l in enumerate(lens):
        out[pg * BLK: pg * BLK + l] = x[int(cu[b]): int(cu[b + 1])]
        table[b, : nblk[b]] = torch.arange(pg, pg + nblk[b], device=x.device, dtype=torch.int32)
        pg += nblk[b]
    return out, table, sum(nblk)


def fork_prefill(ci, cw, q, k, v, iq, ik, lens: list[int]):
    """SGLang prefill for packed docs: returns (topk [HI, T, K], gathered out, paged out, fp32 scores, pages)."""
    cu = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(), device=q.device, dtype=torch.int32)
    lens_t = torch.tensor(lens, device=q.device, dtype=torch.int32)
    T, max_len = int(cu[-1]), max(lens)
    ik_pages, table, total_pages = paged(ik, lens, cu)
    ws = ci.prepare_m3_prefill_indexer(cu, lens_t, total_q=T, max_seqlen_q=max_len, max_seqlen_k=max_len, topk=K, num_index_heads=HI)
    topk = ci.minimax_m3_cute_prefill_indexer_topk(iq, ik_pages, table, cu, lens_t, max_seqlen_q=max_len, max_seqlen_k=max_len, topk=K,
                                                    init_blocks=0, local_blocks=1, sm_scale=SCALE, workspace=ws).clone()
    gathered, _, _ = cw.cute_topk_sparse_attention(q, k, v, topk, cu, cu, max_len, max_len, BLK, K, SCALE)
    kc, _, _ = paged(k, lens, cu)
    vc, _, _ = paged(v, lens, cu)
    out_paged, _, _ = cw.cute_paged_topk_sparse_attention(q, kc, vc, topk, table, lens_t, cu, cu, max_len, max_len, BLK, K, BLK, SCALE,
                                                          total_seqblocks_k=total_pages)
    return topk, gathered, out_paged, ws.score, total_pages
