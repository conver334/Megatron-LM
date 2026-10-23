# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE layer shown as SGLang's Triton fused-MoE path computes it (deterministic inference, EP=1 numerics):
routed experts through the vendored ``fused_moe_kernel`` ×2 with the shared Triton swigluoai in between, routed
weights applied in fp32 inside the down GEMM, slot-ordered fp32 ``moe_sum_reduce``; the shared expert either
separate (``final + shared`` in bf16, what SGLang does with EP>1 / --disable-shared-experts-fusion) or fused as the
extra routing slot (SGLang's EP=1 default). lite's dispatcher + TE grouped GEMM graph stays the backward."""

from __future__ import annotations

import torch

from megatron.lite.model.minimax_m3.lite.model import M3MoELayer
from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
from megatron.lite.primitive.kernels.sgl_fused_moe import fused_experts_bf16
from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate
from megatron.lite.primitive.kernels.sgl_swiglu import swigluoai
from megatron.lite.primitive.kernels.visible import visible_forward

SHARED_MODES = ("separate", "fused")


class SglM3MoELayer(M3MoELayer):
    align_moe_shared: str = "separate"
    _routing = None  # (scores fp32 [T, K], ids [T, K]) of the last router call (SGLang slot order with the router switch)
    _logits = None  # fp32 router logits of the last call (fused shared-expert slot needs the K+1 routing)

    @classmethod
    def swap_in(cls, module: M3MoELayer, *, shared: str = "separate") -> "SglM3MoELayer":
        if shared not in SHARED_MODES:
            raise ValueError(f"align experts shared={shared!r}; known: {SHARED_MODES}")
        if module.dispatcher.ep_size != 1 or module.experts.etp_group is not None:
            raise NotImplementedError("align experts: ep=1 only (the EP combine contract is not fixed yet)")
        module.__class__ = cls
        module.align_moe_shared = shared
        module.router.register_forward_hook(lambda m, inp, out: setattr(module, "_routing", (out[0].detach(), out[1].detach())))
        module.router.gate.register_forward_hook(lambda m, inp, out: setattr(module, "_logits", out.detach()))
        return module

    def _stacked_expert_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """[E, 2I, H] and [E, H, I] from TE GroupedLinear's per-expert parameters (a copy; C10 item)."""
        n = self.experts.num_local_experts
        w13 = torch.stack([getattr(self.experts.fc1, f"weight{e}").detach() for e in range(n)])
        w2 = torch.stack([getattr(self.experts.fc2, f"weight{e}").detach() for e in range(n)])
        return w13, w2

    def _visible(self, x2: torch.Tensor) -> torch.Tensor:
        cfg = self.shared_expert.config
        alpha, limit = cfg.swiglu_alpha, cfg.swiglu_limit
        scores, ids = self._routing
        w13, w2 = self._stacked_expert_weights()
        w13_s = self.shared_expert.gate_up.linear.weight.detach()
        w2_s = self.shared_expert.down.linear.weight.detach()
        if self.align_moe_shared == "separate":
            routed = fused_experts_bf16(x2, w13, w2, scores, ids, alpha, limit)
            shared = linear_bf16(swigluoai(linear_bf16(x2, w13_s), alpha, limit), w2_s)
            return routed + shared
        k = ids.shape[1]
        weights, slots = moe_fused_gate(
            self._logits, self.router.expert_bias, k + 1, num_fused_shared_experts=1, renormalize=True,
            routed_scaling_factor=self.router.scaling_factor or 1.0, apply_routed_scaling_factor_on_output=True,
        )
        return fused_experts_bf16(x2, torch.cat([w13, w13_s[None]]), torch.cat([w2, w2_s[None]]), weights, slots, alpha, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native = super().forward(x)  # runs the router (hooks capture its outputs) and lite's dispatcher/experts graph
        with torch.no_grad():
            visible = self._visible(x.reshape(-1, x.size(-1))).view(x.shape).to(x.dtype)
        return visible_forward(visible, native)
