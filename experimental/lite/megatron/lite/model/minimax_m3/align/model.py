# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 align model: ``lite`` layers with aligned modules swapped in per ``train_config.align_modules``.

Each entry of ``align_modules`` names one module class whose forward is replaced by the SGLang kernel
(backward keeps the lite/TE implementation). The registry below maps the name to a swap function
``(layer, config, ps) -> None``; it is filled in module by module as the C5 gate progresses.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from megatron.lite.primitive.kernels.visible import visible_forward

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Layer, MiniMaxM3Model
from megatron.lite.primitive.parallel import ParallelState

Swap = Callable[..., None]  # (owner, config, ps, **align_opts)



def _swap_rmsnorm(owner: nn.Module, config: MiniMaxM3Config, ps: ParallelState, **_opts) -> None:
    """MoE ``mlp_norm`` and the final ``norm`` hold their weight in ``SglGemmaRMSNorm``; the align forward feeds
    them through SGLang's fused add+norm (``MiniMaxM3AlignLayer.forward`` / ``MiniMaxM3AlignModel._final_norm``).
    Norms fused into LayerNormLinear (attention input, dense MLP input) are not touched yet."""
    from megatron.lite.primitive.kernels.sgl_norm import SglGemmaRMSNorm

    for name in ("mlp_norm", "norm"):
        old = getattr(owner, name, None)
        if old is not None:
            setattr(owner, name, SglGemmaRMSNorm.from_te(old, eps=config.rms_norm_eps))


def _swap_router(owner: nn.Module, config: MiniMaxM3Config, ps: ParallelState, *, router_gate: str = "fp32", **_opts) -> None:
    """MoE router -> SGLang gate GEMM dtype + moe_fused_gate selection (slot order, lowest-id ties)."""
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    moe = getattr(owner, "moe", None)
    if moe is not None:
        SglSigmoidTopKRouter.swap_in(moe.router, gate=router_gate)


def _swap_o_proj(owner: nn.Module, config: MiniMaxM3Config, ps: ParallelState, **_opts) -> None:
    """Attention output projection -> SGLang batch-invariant bf16 GEMM (DeepGEMM) forward, TE backward."""
    from megatron.lite.primitive.parallel.linear_sgl import SglRowParallelLinear

    attn = getattr(owner, "attn", None)
    if attn is not None:
        SglRowParallelLinear.swap_in(attn.proj)


def _swap_qkv(owner: nn.Module, config: MiniMaxM3Config, ps: ParallelState, **_opts) -> None:
    """Attention input norm + QKV projection -> SGLang fused add+norm of the residual pair + DeepGEMM; TE backward."""
    from megatron.lite.primitive.parallel.linear_sgl import SglColumnParallelLinear

    attn = getattr(owner, "attn", None)
    if attn is not None:
        SglColumnParallelLinear.swap_in(attn.qkv)


def _swap_dense_mlp(owner: nn.Module, config: MiniMaxM3Config, ps: ParallelState, **_opts) -> None:
    """Dense-layer MLP -> SGLang fused add+norm + DeepGEMM gate_up/down + Triton swigluoai; TE backward."""
    from megatron.lite.model.minimax_m3.align.dense_mlp import SglM3DenseMLP

    mlp = getattr(owner, "mlp", None)
    if mlp is not None:
        SglM3DenseMLP.swap_in(mlp)


def _swap_lm_head(owner: nn.Module, config: MiniMaxM3Config, ps: ParallelState, **_opts) -> None:
    """Model head -> batch-invariant bf16 GEMM; token loss -> -logprob from SGLang's BI log_softmax."""
    from megatron.lite.model.minimax_m3.align.lm_head import SglVocabParallelOutput

    head = getattr(owner, "head", None)
    if head is not None:
        SglVocabParallelOutput.swap_in(head)


def _swap_experts(owner: nn.Module, config: MiniMaxM3Config, ps: ParallelState, *, moe_shared: str = "separate", **_opts) -> None:
    """MoE layer -> SGLang Triton fused-MoE bytes (deterministic config), shared expert separate or fused slot."""
    from megatron.lite.model.minimax_m3.align.moe import SglM3MoELayer

    moe = getattr(owner, "moe", None)
    if moe is not None:
        SglM3MoELayer.swap_in(moe, shared=moe_shared)


# name -> swap, applied to every layer and then to the model (final norm / head live there);
# keep the keys identical to the C0 inventory rows of docs/train_infer_consistency_plan.md
ALIGN_SWAPS: dict[str, Swap] = {
    "rmsnorm": _swap_rmsnorm,
    "router": _swap_router,
    "o_proj": _swap_o_proj,
    "qkv": _swap_qkv,
    "dense_mlp": _swap_dense_mlp,
    "lm_head": _swap_lm_head,
    "experts": _swap_experts,
}


class MiniMaxM3AlignLayer(MiniMaxM3Layer):
    def __init__(self, config: MiniMaxM3Config, ps: ParallelState, layer_idx: int, *, align_modules=(), align_opts=None, **kwargs):
        super().__init__(config, ps, layer_idx, **kwargs)
        unknown = [m for m in align_modules if m not in ALIGN_SWAPS]
        if unknown:
            raise ValueError(f"minimax_m3 align: unknown align_modules {unknown}; known: {sorted(ALIGN_SWAPS)}")
        for name in ALIGN_SWAPS:  # registry order: router before experts (the MoE swap needs the SGLang router)
            if name in align_modules:
                ALIGN_SWAPS[name](self, config, ps, **(align_opts or {}))
        self.align_modules = tuple(align_modules)
        self._residual_pair = None  # (delta, residual) of this layer's last add; consumed by the next norm site
        self._prev = None  # previous align layer on this stage (set by MiniMaxM3AlignModel)

    def forward(self, x: torch.Tensor, magi_ctx):
        """lite's layer with SGLang's norm sites: every norm that follows a residual add sees the fp32 sum of the
        (delta, residual) pair (kernel bytes) while the native TE/torch path of the bf16 sum carries the gradient.
        Pairs are plain attributes that stay set until the next forward, so activation recompute replays them."""
        fused_norms = "rmsnorm" in self.align_modules
        if not fused_norms and not ({"qkv", "dense_mlp"} & set(self.align_modules)):
            return super().forward(x, magi_ctx)
        from megatron.lite.primitive.kernels.sgl_norm import visible_input_norm

        if "qkv" in self.align_modules:  # SGLang input_layernorm(prev_out, residual) inside the swapped qkv
            self.attn.qkv._residual_pair = self._prev._residual_pair if self._prev is not None else None
        if self.is_sparse_attention:
            delta = self.attn(x, magi_ctx=magi_ctx)
        else:
            delta = self.attn(
                x, position_ids=magi_ctx.position_ids, packed_seq_params=magi_ctx.dense_packed_seq_params()
            )
        pair = (delta, x)
        x = x + delta  # bf16 sum == the residual the fused kernel writes back
        if self.moe is not None:
            native = self.mlp_norm(x)  # module call: DDP param-gather pre-hook, TE/standalone backward
            normed = visible_forward(visible_input_norm(x, self.mlp_norm.weight, self.mlp_norm.eps, pair), native) if fused_norms else native
            delta = self.moe(normed)
        else:
            if "dense_mlp" in self.align_modules:  # SGLang post_attention_layernorm(attn_out, residual) inside the MLP
                self.mlp._residual_pair = pair
            delta = self.mlp(x)
        self._residual_pair = (delta, x)  # the next input norm / final norm fuses this add the same way
        return x + delta


class MiniMaxM3AlignModel(MiniMaxM3Model):
    def __init__(self, config: MiniMaxM3Config, train_config, ps: ParallelState, *, vpp_chunk_id: int | None = None):
        align_modules = tuple(getattr(train_config, "align_modules", ()))
        align_opts = dict(getattr(train_config, "align_opts", None) or {})
        self.layer_cls = _layer_factory(align_modules, align_opts)
        super().__init__(config, train_config, ps, vpp_chunk_id=vpp_chunk_id)
        for name in ALIGN_SWAPS:
            if name in align_modules:
                ALIGN_SWAPS[name](self, config, ps, **align_opts)
        self.align_modules = align_modules
        for prev, layer in zip(self.layers, list(self.layers)[1:]):
            object.__setattr__(layer, "_prev", prev)  # plain reference, not a registered submodule

    def forward(self, *args, **kwargs) -> dict:
        for layer in self.layers:  # drop last step's pairs now; they must survive backward/recompute of that step
            layer._residual_pair = None
            for name in ("qkv", "dense_mlp"):
                owner = layer.attn.qkv if name == "qkv" else layer.mlp
                if name in self.align_modules and owner is not None:
                    owner._residual_pair = None
        return super().forward(*args, **kwargs)

    def _token_loss(self, logits: torch.Tensor, labels_sb: torch.Tensor) -> torch.Tensor:
        if "lm_head" not in self.align_modules:
            return super()._token_loss(logits, labels_sb)
        from megatron.lite.model.minimax_m3.align.lm_head import sgl_token_loss

        return sgl_token_loss(logits, labels_sb, self.ps.tp_group)

    def _final_norm(self, h: torch.Tensor) -> torch.Tensor:
        native = self.norm(h)  # module call (DDP pre-hook); gradient of lite's norm of the bf16 sum
        if "rmsnorm" not in self.align_modules:
            return native
        from megatron.lite.primitive.kernels.sgl_norm import visible_input_norm

        return visible_forward(visible_input_norm(h, self.norm.weight, self.norm.eps, self.layers[-1]._residual_pair), native)


def _layer_factory(align_modules: tuple[str, ...], align_opts: dict | None = None):
    def make(config, ps, layer_idx, **kwargs) -> nn.Module:
        return MiniMaxM3AlignLayer(config, ps, layer_idx, align_modules=align_modules, align_opts=align_opts, **kwargs)

    return make
