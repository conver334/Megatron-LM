# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's FA4 forward (CuTe DSL) with its batch-invariant switch, vendored from the fork
conver334/sglang@291de67e3d kernels/ops/attention/flash_attn/cute (Tri Dao's flash-attention cute package,
BSD-3, see cute/LICENSE; SGLang additions Apache-2.0). Only the module paths and three SGLang helper imports were
rewritten (_compat). The batch-invariant variant masks every KV block (block_info), which is the only
variant whose rows agree between chunked and full prefill; pip flash_attn.cute equals the switch-off variant."""

from __future__ import annotations

import torch


def flash_attn_varlen_bi(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor,
                         softmax_scale: float, max_seqlen: int | None = None) -> torch.Tensor:
    """Causal varlen attention as SGLang's deterministic mode runs it: batch-invariant kernel, num_splits=1."""
    from megatron.lite.primitive.kernels.sgl_fa4.cute.batch_invariance import set_batch_invariant
    from megatron.lite.primitive.kernels.sgl_fa4.cute.interface import flash_attn_varlen_func

    set_batch_invariant(True)
    if max_seqlen is None:
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen,
                                 max_seqlen_k=max_seqlen, softmax_scale=softmax_scale, causal=True, num_splits=1)
    return out[0] if isinstance(out, tuple) else out
