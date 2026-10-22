# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 align model: ``lite`` layers with aligned modules swapped in per ``train_config.align_modules``.

Each entry of ``align_modules`` names one module class whose forward is replaced by the SGLang kernel
(backward keeps the lite/TE implementation). The registry below maps the name to a swap function
``(layer, config, ps) -> None``; it is filled in module by module as the C5 gate progresses.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Layer, MiniMaxM3Model
from megatron.lite.primitive.parallel import ParallelState

Swap = Callable[..., None]  # (owner, config, ps, **align_opts)



def _swap_rmsnorm(owner: nn.Module, config: MiniMaxM3Config, ps: ParallelState, **_opts) -> None:
    """Standalone norms (MoE ``mlp_norm``, final ``norm``) -> SGLang gemma_rmsnorm forward.
    Norms fused into LayerNormLinear (attention input, dense MLP input) are not touched yet."""
    from megatron.lite.primitive.kernels.sgl_norm import SglGemmaRMSNorm

    for name in ("mlp_norm", "norm"):
        old = getattr(owner, name, None)
        if old is not None:
            setattr(owner, name, SglGemmaRMSNorm.from_te(old, eps=config.rms_norm_eps))


def _swap_router(owner: nn.Module, config: MiniMaxM3Config, ps: ParallelState, *, router_gate: str = "bf16", **_opts) -> None:
    """MoE router -> SGLang gate GEMM dtype + moe_fused_gate selection (slot order, lowest-id ties)."""
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    moe = getattr(owner, "moe", None)
    if moe is not None:
        SglSigmoidTopKRouter.swap_in(moe.router, gate=router_gate)


# name -> swap, applied to every layer and then to the model (final norm / head live there);
# keep the keys identical to the C0 inventory rows of docs/train_infer_consistency_plan.md
ALIGN_SWAPS: dict[str, Swap] = {"rmsnorm": _swap_rmsnorm, "router": _swap_router}


class MiniMaxM3AlignLayer(MiniMaxM3Layer):
    def __init__(self, config: MiniMaxM3Config, ps: ParallelState, layer_idx: int, *, align_modules=(), align_opts=None, **kwargs):
        super().__init__(config, ps, layer_idx, **kwargs)
        unknown = [m for m in align_modules if m not in ALIGN_SWAPS]
        if unknown:
            raise ValueError(f"minimax_m3 align: unknown align_modules {unknown}; known: {sorted(ALIGN_SWAPS)}")
        for name in align_modules:
            ALIGN_SWAPS[name](self, config, ps, **(align_opts or {}))
        self.align_modules = tuple(align_modules)


class MiniMaxM3AlignModel(MiniMaxM3Model):
    def __init__(self, config: MiniMaxM3Config, train_config, ps: ParallelState, *, vpp_chunk_id: int | None = None):
        align_modules = tuple(getattr(train_config, "align_modules", ()))
        align_opts = dict(getattr(train_config, "align_opts", None) or {})
        self.layer_cls = _layer_factory(align_modules, align_opts)
        super().__init__(config, train_config, ps, vpp_chunk_id=vpp_chunk_id)
        for name in align_modules:
            ALIGN_SWAPS[name](self, config, ps, **align_opts)
        self.align_modules = align_modules


def _layer_factory(align_modules: tuple[str, ...], align_opts: dict | None = None):
    def make(config, ps, layer_idx, **kwargs) -> nn.Module:
        return MiniMaxM3AlignLayer(config, ps, layer_idx, align_modules=align_modules, align_opts=align_opts, **kwargs)

    return make
