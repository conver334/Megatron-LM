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

Swap = Callable[[MiniMaxM3Layer, MiniMaxM3Config, ParallelState], None]

# name -> swap; keep the keys identical to the C0 inventory rows of docs/train_infer_consistency_plan.md
ALIGN_SWAPS: dict[str, Swap] = {}


class MiniMaxM3AlignLayer(MiniMaxM3Layer):
    def __init__(self, config: MiniMaxM3Config, ps: ParallelState, layer_idx: int, *, align_modules=(), **kwargs):
        super().__init__(config, ps, layer_idx, **kwargs)
        unknown = [m for m in align_modules if m not in ALIGN_SWAPS]
        if unknown:
            raise ValueError(f"minimax_m3 align: unknown align_modules {unknown}; known: {sorted(ALIGN_SWAPS)}")
        for name in align_modules:
            ALIGN_SWAPS[name](self, config, ps)
        self.align_modules = tuple(align_modules)


class MiniMaxM3AlignModel(MiniMaxM3Model):
    def __init__(self, config: MiniMaxM3Config, train_config, ps: ParallelState, *, vpp_chunk_id: int | None = None):
        align_modules = tuple(getattr(train_config, "align_modules", ()))
        self.layer_cls = _layer_factory(align_modules)
        super().__init__(config, train_config, ps, vpp_chunk_id=vpp_chunk_id)


def _layer_factory(align_modules: tuple[str, ...]):
    def make(config, ps, layer_idx, **kwargs) -> nn.Module:
        return MiniMaxM3AlignLayer(config, ps, layer_idx, align_modules=align_modules, **kwargs)

    return make
