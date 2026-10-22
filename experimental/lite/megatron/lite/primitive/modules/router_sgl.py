# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sigmoid top-k router whose forward reproduces SGLang bytes (gate GEMM as ``torch.mm`` like
``_compute_router_logits``, selection/weights from the vendored ``moe_fused_gate`` kernel) and whose
backward follows the routing formula. Applied by class swap onto a built ``SigmoidTopKRouter`` so
parameter names (``gate.weight``, ``expert_bias``) and checkpoint mapping stay unchanged."""

from __future__ import annotations

import torch

from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate
from megatron.lite.primitive.modules.router import SigmoidTopKRouter, _GatingLinear
from megatron.lite.primitive.modules.router_replay import gather_replayed_router_scores


class _GateMM(torch.autograd.Function):
    """SGLang ``_compute_router_logits`` without batch-invariant mode (cuBLAS):
    bf16: ``torch.mm(x, W.bf16.T, out_dtype=fp32)`` (SGLANG_OPT_USE_BF16_ROUTER_GEMM=1, default);
    fp32: ``torch.mm(x.float(), W.float().T)`` (env off). Under --enable-deterministic-inference SGLang
    replaces both with batch-invariant GEMMs (bf16-rounded logits for bf16 gate); not reproduced here yet.
    dX uses the operand the forward used (bf16(W) in bf16 mode), like RouterGatingLinearFunction."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, gate_dtype: torch.dtype) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.gate_dtype = gate_dtype
        if gate_dtype == torch.bfloat16:
            return torch.mm(x.to(torch.bfloat16), weight.to(torch.bfloat16).t(), out_dtype=torch.float32)
        return torch.mm(x.float(), weight.float().t())

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x, weight = ctx.saved_tensors
        g = grad.float()
        dx = torch.mm(g, weight.to(ctx.gate_dtype).float()).to(x.dtype)
        return dx, torch.mm(g.t(), x.float()).to(weight.dtype), None


class _SglGatingLinear(_GatingLinear):
    """Owns ``gate.weight``: the GEMM must run inside this module's forward so the DDP param-gather pre-hook fires."""

    def forward(self, x: torch.Tensor, gate_dtype: torch.dtype) -> torch.Tensor:  # type: ignore[override]
        return _GateMM.apply(x, self.weight, gate_dtype)


class SglSigmoidTopKRouter(SigmoidTopKRouter):
    align_gate_dtype: torch.dtype = torch.bfloat16

    @classmethod
    def swap_in(cls, router: SigmoidTopKRouter, *, gate: str = "bf16") -> "SglSigmoidTopKRouter":
        if router.score_function != "sigmoid" or router.num_groups is not None:
            raise NotImplementedError("align router: sigmoid, ungrouped routing only")
        router.__class__ = cls
        router.gate.__class__ = _SglGatingLinear
        router.align_gate_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[gate]
        return router

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x.reshape(-1, x.size(-1)), self.align_gate_dtype)
        weights_k, ids_k = moe_fused_gate(
            logits.detach(),
            self.expert_bias,
            self.topk,
            renormalize=True,
            routed_scaling_factor=self.scaling_factor or 1.0,
            apply_routed_scaling_factor_on_output=True,
        )
        ids = ids_k.long()  # SGLang slot order (biased-descending), not expert-id order
        sel = torch.gather(torch.sigmoid(logits), 1, ids)
        probs = sel / sel.sum(dim=-1, keepdim=True) * (self.scaling_factor or 1.0)
        scores = probs + (weights_k - probs).detach()  # kernel bytes forward, formula backward
        if self.router_replay is not None:
            selected = self.router_replay.select_indices(ids)
            if selected is not ids:
                # replay returns a fresh tensor even when nothing changed: keep kernel bytes on unchanged rows
                replayed = gather_replayed_router_scores(
                    logits, selected, score_function=self.score_function, scaling_factor=self.scaling_factor
                ).to(logits.dtype)
                same = (selected == ids).all(dim=-1, keepdim=True)
                return torch.where(same, scores, replayed), selected
        return scores, ids
