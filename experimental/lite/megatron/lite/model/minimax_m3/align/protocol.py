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
    # router gate GEMM as in SGLang (C1 decision 10-22: inference runs SGLANG_OPT_USE_BF16_ROUTER_GEMM=0, fp32 gate):
    # "fp32" = cuBLAS torch.mm (BI=0 baseline); "fp32_deepgemm" = SGLANG_OPT_BF16_FP32_GEMM_ALGO=deep_gemm, the
    # batch-invariant contract (bf16 high/low split through DeepGEMM); "bf16" = the SGLang default we do not use.
    align_router_gate: str = "fp32"
    # shared expert in the MoE switch: "separate" (SGLang EP>1 / --disable-shared-experts-fusion: final + shared in
    # bf16) or "fused" (SGLang EP=1 default: extra routing slot, summed in the fp32 moe_sum_reduce)
    align_moe_shared: str = "separate"
    # slot accumulation order of the MoE combine: "slot" (SGLang EP=1 Triton moe_sum_reduce), "expert_asc"
    # (ascending expert id, the EP contract candidate), "ep_normal" (DeepEP normal-mode shape for align_moe_ep_size ranks)
    align_moe_combine: str = "slot"
    align_moe_ep_size: int = 1
    # expert kernels: "triton" (SGLang EP=1 fused_moe) or "deepgemm_ep" (SGLang DeepEP normal + DeepGEMM bf16 runner
    # over align_moe_ep_size >= 2 ranks: ep_scatter layout, m-grouped GEMMs, ep_gather, DeepEP combine; the combine
    # order is DeepEP's, align_moe_combine must stay "slot" or "ep_normal")
    align_moe_experts: str = "triton"
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

    if "qkv" in impl_cfg.align_modules and (impl_cfg.parallel.pp > 1 or impl_cfg.parallel.vpp > 1):
        raise NotImplementedError("align qkv: the residual pair is not carried across pipeline stages yet (pp=1, vpp=1)")
    bundle = _lite.build_model(
        model_cfg,
        impl_cfg=to_lite_impl_cfg(impl_cfg),
        model_cls=MiniMaxM3AlignModel,
        extra_train_cfg={
            "align_modules": tuple(impl_cfg.align_modules),
            "align_opts": {"router_gate": impl_cfg.align_router_gate, "moe_shared": impl_cfg.align_moe_shared,
                           "moe_combine": impl_cfg.align_moe_combine, "moe_ep_size": impl_cfg.align_moe_ep_size,
                           "moe_experts": impl_cfg.align_moe_experts},
        },
    )
    for chunk in bundle.chunks:
        chunk.align_roll_labels = impl_cfg.roll_labels
    return replace(bundle, forward_step=_forward_step)
