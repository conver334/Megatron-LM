# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense-layer core attention shown as SGLang's deterministic FA4 computes it (batch-invariant kernel, num_splits=1);
pip FA4 (forward + backward) stays the training graph."""

from __future__ import annotations

import torch

from megatron.lite.primitive.kernels.sgl_fa4 import flash_attn_varlen_bi
from megatron.lite.primitive.kernels.visible import visible_forward
from megatron.lite.primitive.modules.attention.magi import MagiDotProductAttention


class SglFa4DotProductAttention(MagiDotProductAttention):
    def forward(self, query, key, value, *, packed_seq_params):
        from megatron.lite.primitive.recompute import is_recomputing

        native = super().forward(query, key, value, packed_seq_params=packed_seq_params)
        fa4_varlen = getattr(packed_seq_params, "fa4_varlen", None)
        if fa4_varlen is None or is_recomputing():
            return native
        cu_seqlens = fa4_varlen[0]
        with torch.no_grad():
            visible = flash_attn_varlen_bi(query.detach(), key.detach(), value.detach().contiguous(), cu_seqlens, self.softmax_scale,
                                           getattr(packed_seq_params, "max_seqlen_q", None))
        return visible_forward(visible.view_as(native), native)

    @classmethod
    def swap_in(cls, core_attn: MagiDotProductAttention) -> "SglFa4DotProductAttention":
        if not isinstance(core_attn, MagiDotProductAttention):
            raise NotImplementedError("align attn_dense needs the magi fa4_varlen core attention")
        core_attn.__class__ = cls
        return core_attn
