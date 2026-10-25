# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""RoPE shown as SGLang applies it (CPU-built fp32 cos/sin table, fp32 rotate, bf16 store) on q, k and the indexer's
index-q/k; TE's fused RoPE stays the backward."""

from __future__ import annotations

import torch

from megatron.lite.primitive.kernels.sgl_rope import CosSinCache, sgl_rope
from megatron.lite.primitive.kernels.visible import visible_forward
from megatron.lite.primitive.modules.attention.msa import MSAIndexer, MSAttention
from megatron.lite.primitive.modules.gqa import GQAttention
from megatron.lite.primitive.utils.rope import apply_rope_sbhd

_CACHES: dict[tuple[float, int], CosSinCache] = {}


def shared_cache(theta: float, rot: int) -> CosSinCache:
    return _CACHES.setdefault((float(theta), int(rot)), CosSinCache(theta, rot))


def _visible_rope(x: torch.Tensor, positions: torch.Tensor, cache: CosSinCache, native: torch.Tensor) -> torch.Tensor:
    from megatron.lite.primitive.recompute import is_recomputing

    if is_recomputing() or positions is None:
        return native
    T, B, H, D = x.shape
    with torch.no_grad():
        pos = positions.reshape(-1).to(device=x.device, dtype=torch.int64)
        vis = sgl_rope(x.detach().reshape(T * B, H, D).contiguous(), pos, cache.rows(pos), cache.rot).view(T, B, H, D)
    return visible_forward(vis, native)


class _SglRope:
    _sgl_cache: CosSinCache

    def _rope(self, x, freqs, positions):
        fused = getattr(self, "_rope_fusion", getattr(self, "rope_fusion", False))
        return _visible_rope(x, positions, self._sgl_cache, apply_rope_sbhd(x, freqs, fused=fused))


class SglRopeGQAttention(_SglRope, GQAttention):
    pass


class SglRopeMSAttention(_SglRope, MSAttention):
    pass


class SglRopeMSAIndexer(_SglRope, MSAIndexer):
    pass


def swap_in(attn: GQAttention, *, theta: float, rot: int) -> None:
    cache = shared_cache(theta, rot)
    attn.__class__ = SglRopeMSAttention if isinstance(attn, MSAttention) else SglRopeGQAttention
    attn._sgl_cache = cache
    indexer = getattr(attn, "indexer", None)
    if indexer is not None:
        indexer.__class__ = SglRopeMSAIndexer
        indexer._sgl_cache = cache
