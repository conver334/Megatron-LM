# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax sparse attention as SGLang's prefill computes it. The fork's CuTe path (``inference.msa_v1`` indexer GEMM +
CUDA radix top-k + ``cute_paged_topk_sparse_attention``) runs the same MM-Sparse-Attention ``msa_v1`` kernels the
training side (Magi) uses; the bytes differ only through the call ABI: inference scores with lse temperature 1 and
runs the attention with ``lse_temperature_scale=1`` (raw partial max/sum statistics), while the MiniMax KL recipe
drives Magi with ``kl_temperature=2`` (scaled scores, normalised partial statistics). Tested bitwise against the fork
chain by path (tests/unit/primitive/kernels/test_sgl_msa_crossstack_gpu.py), so this module calls the training kernels
with the inference ABI instead of carrying the fork glue."""

from __future__ import annotations

import importlib

import torch

HEAD_DIM, Q_HEADS, KV_HEADS, INDEX_HEADS, TOPK, BLOCK = 128, 64, 4, 4, 16, 128


def sgl_msa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, index_q: torch.Tensor, index_k: torch.Tensor,
                    cu_seqlens: list[int]) -> torch.Tensor:
    """Packed documents in order: q [T, 64, 128], k/v [T, 4, 128], index_q [T, 4, 128], index_k [T, 1, 128] bf16,
    ``cu_seqlens`` host ints -> sparse attention output [T, 64, 128] bf16 with SGLang's bytes."""
    indexer = importlib.import_module("msa_v1.indexer")
    attention = importlib.import_module("msa_v1.attention")
    lens = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]
    cu = torch.tensor(cu_seqlens, device=q.device, dtype=torch.int32)
    max_len = max(lens)
    total = int(cu[-1])
    pages = sum((l + BLOCK - 1) // BLOCK for l in lens)
    topk, _ = indexer.forward(index_q.contiguous(), index_k.contiguous(), cu_seqlens_q=cu, cu_seqlens_kv=cu,
                              max_seqlen_q=max_len, max_seqlen_kv=max_len, deterministic=True)  # lse temperature 1
    meta = attention.prepare(topk, cu, cu, total_k=total, total_rows=pages, max_seqlen_q=max_len, max_seqlen_k=max_len,
                             fragment_indices=None, prepare_kl_schedule=False)
    out, _ = attention.forward(q.contiguous(), k.contiguous(), v.contiguous(), meta, softmax_scale=HEAD_DIM ** -0.5,
                               return_softmax_lse=True, partial_dtype=torch.bfloat16)  # lse_temperature_scale 1: raw stats
    return out
