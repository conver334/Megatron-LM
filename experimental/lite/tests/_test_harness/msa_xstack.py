# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared glue for the MSA cross-stack checks: SGLang's prefill chain (CuTe indexer + radix top-k + paged sparse
attention) run by path on packed documents given as flat [T, heads, 128] bf16 tensors."""

from __future__ import annotations

import torch

HQ, HKV, D, HI, K, BLK = 64, 4, 128, 4, 16, 128
SCALE = D ** -0.5


def paged(x: torch.Tensor, lens: list[int], cu: torch.Tensor):
    """Per-document pages of BLK tokens (SGLang's page128 KV layout) and the page table (invalid pages 0, as the
    fork's build_cute_page_table writes them)."""
    nblk = [(l + BLK - 1) // BLK for l in lens]
    out = torch.zeros(sum(nblk) * BLK, *x.shape[1:], device=x.device, dtype=x.dtype)
    table = torch.zeros((len(lens), max(nblk)), device=x.device, dtype=torch.int32)
    pg = 0
    for b, l in enumerate(lens):
        out[pg * BLK: pg * BLK + l] = x[int(cu[b]): int(cu[b + 1])]
        table[b, : nblk[b]] = torch.arange(pg, pg + nblk[b], device=x.device, dtype=torch.int32)
        pg += nblk[b]
    return out, table, sum(nblk)


def fork_prefill(ci, cw, q, k, v, iq, ik, lens: list[int], prefix_lens: list[int] | None = None):
    """SGLang prefill for packed docs. ``k``/``v``/``ik`` hold every token of each document (prefix + new, in order);
    ``q``/``iq`` only the new tokens (``lens``); ``prefix_lens`` > 0 is a chunked-prefill extend over cached KV.
    Returns (topk [HI, T_new, K], gathered out (prefix 0 only, else None), paged out, fp32 scores, pages)."""
    prefix = prefix_lens or [0] * len(lens)
    seq = [p + l for p, l in zip(prefix, lens)]
    cu = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(), device=q.device, dtype=torch.int32)
    cu_k = torch.tensor([0] + torch.cumsum(torch.tensor(seq), 0).tolist(), device=q.device, dtype=torch.int32)
    seq_t = torch.tensor(seq, device=q.device, dtype=torch.int32)
    T, max_q, max_k = int(cu[-1]), max(lens), max(seq)
    ik_pages, table, total_pages = paged(ik, seq, cu_k)
    ws = ci.prepare_m3_prefill_indexer(cu, seq_t, total_q=T, max_seqlen_q=max_q, max_seqlen_k=max_k, topk=K, num_index_heads=HI)
    topk = ci.minimax_m3_cute_prefill_indexer_topk(iq, ik_pages, table, cu, seq_t, max_seqlen_q=max_q, max_seqlen_k=max_k, topk=K,
                                                    init_blocks=0, local_blocks=1, sm_scale=SCALE, workspace=ws).clone()
    gathered = None
    if not any(prefix):
        gathered, _, _ = cw.cute_topk_sparse_attention(q, k, v, topk, cu, cu, max_q, max_k, BLK, K, SCALE)
    kc, _, _ = paged(k, seq, cu_k)
    vc, _, _ = paged(v, seq, cu_k)
    out_paged, _, _ = cw.cute_paged_topk_sparse_attention(q, kc, vc, topk, table, seq_t, cu, cu_k, max_q, max_k, BLK, K, BLK, SCALE,
                                                          total_seqblocks_k=total_pages)
    return topk, gathered, out_paged, ws.score, total_pages
