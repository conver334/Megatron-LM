# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime protocol for the MiniMax-M3 ``align`` implementation (runtime key ``minimax_m3_align``).

Wraps ``lite.protocol``: same config, checkpoint mapping, forward step and optimizers; defaults follow
``MINIMAX_TRAINING_IMPL`` and ``align_modules`` selects which modules call the SGLang kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
import torch.nn as nn

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
from megatron.lite.model.minimax_m3.lite import protocol as _lite
from megatron.lite.model.minimax_m3.lite.protocol import (  # noqa: F401  (re-exported protocol surface)
    EXPERT_CLASSIFIER,
    MINIMAX_TRAINING_IMPL,
    MODULE_MAP,
    PLACEMENT_FN,
    build_model_config,
    export_hf_weights,
    load_hf_weights,
    save_hf_weights,
    unpack_forward_output,
    vocab_size,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.runtime.contracts.data import PackedBatch

__all__ = [
    "EXPERT_CLASSIFIER",
    "ImplConfig",
    "PLACEMENT_FN",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
    "vocab_size",
]



@dataclass(frozen=True)
class ImplConfig(_lite.ImplConfig):
    # MiniMax training recipe as defaults (fa4_varlen requires cp=1)
    rope_fusion: bool = MINIMAX_TRAINING_IMPL["rope_fusion"]
    msa_index_input_norm: str = MINIMAX_TRAINING_IMPL["msa_index_input_norm"]
    msa_indexer_kl_loss_coeff: float = MINIMAX_TRAINING_IMPL["msa_indexer_kl_loss_coeff"]
    msa_indexer_kl_temperature: float = MINIMAX_TRAINING_IMPL["msa_indexer_kl_temperature"]
    magi_dense_kernel_backend: str = MINIMAX_TRAINING_IMPL["magi_dense_kernel_backend"]
    moe_permute_fusion: bool = MINIMAX_TRAINING_IMPL["moe_permute_fusion"]
    # modules whose forward calls the SGLang kernel; () == lite numerics (see align/model.py ALIGN_SWAPS)
    align_modules: tuple[str, ...] = ()
    # router gate GEMM as in SGLang: "bf16" = SGLANG_OPT_USE_BF16_ROUTER_GEMM=1 (default), "fp32" = env off
    align_router_gate: str = "bf16"
    # PackedBatch contract (verl_mlite, other protocols): labels/loss_mask arrive unrolled and the protocol rolls
    # them left per document (last token masked). lite does not roll; keep False only for lite-style batches.
    roll_labels: bool = True


def to_lite_impl_cfg(impl_cfg: ImplConfig) -> _lite.ImplConfig:
    # getattr, not asdict: asdict would flatten ParallelConfig/OptimizerConfig into dicts
    values = {f.name: getattr(impl_cfg, f.name) for f in fields(_lite.ImplConfig)}
    # verl_mlite forces use_thd=True ("no-padding batches"); M3 consumes PackedBatch natively through Magi,
    # so the flag is satisfied by construction and must not trip lite's NotImplementedError.
    values["use_thd"] = False
    return _lite.ImplConfig(**values)


def roll_packed_labels(batch: PackedBatch) -> PackedBatch:
    """Shift ``labels``/``loss_mask`` left by one inside every document and mask the last token
    (same convention as ``primitive.parallel.thd.pack_nested_thd(roll_labels=True)``)."""
    seq_lens = [int(n) for n in batch.seq_lens.tolist()]

    def _roll(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        out = t.reshape(-1).clone()
        start = 0
        for n in seq_lens:
            if n > 0:
                out[start : start + n] = torch.roll(out[start : start + n], shifts=-1, dims=0)
                out[start + n - 1] = 0
            start += n
        return out

    return replace(batch, labels=_roll(batch.labels), loss_mask=_roll(batch.loss_mask))


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    if getattr(_lite._unwrap(model), "align_roll_labels", False):
        batch = roll_packed_labels(batch)
    return _lite._forward_step(model, batch)


def build_model(model_cfg: MiniMaxM3Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    from megatron.lite.model.minimax_m3.align.model import MiniMaxM3AlignModel

    bundle = _lite.build_model(
        model_cfg,
        impl_cfg=to_lite_impl_cfg(impl_cfg),
        model_cls=MiniMaxM3AlignModel,
        extra_train_cfg={
            "align_modules": tuple(impl_cfg.align_modules),
            "align_opts": {"router_gate": impl_cfg.align_router_gate},
        },
    )
    for chunk in bundle.chunks:
        chunk.align_roll_labels = impl_cfg.roll_labels
    return replace(bundle, forward_step=_forward_step)
