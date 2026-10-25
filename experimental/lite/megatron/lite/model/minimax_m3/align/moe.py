# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE layer shown as SGLang's Triton fused-MoE path computes it (deterministic inference, EP=1 numerics):
routed experts through the vendored ``fused_moe_kernel`` ×2 with the shared Triton swigluoai in between, routed
weights applied in fp32 inside the down GEMM, slot-ordered fp32 ``moe_sum_reduce``; the shared expert either
separate (``final + shared`` in bf16, what SGLang does with EP>1 / --disable-shared-experts-fusion) or fused as the
extra routing slot (SGLang's EP=1 default); the slot accumulation order is a declared contract option (``combine``:
slot = SGLang EP=1, expert_asc, ep_normal). ``experts="deepgemm_ep"`` instead shows SGLang's DeepEP-normal + DeepGEMM
bf16 path over ``ep_size`` ranks (sgl_deepgemm_moe). lite's dispatcher + TE grouped GEMM graph stays the backward."""

from __future__ import annotations

import torch

from megatron.lite.model.minimax_m3.lite.model import M3MoELayer
from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
from megatron.lite.primitive.kernels.sgl_deepgemm_moe import fused_experts_deepgemm_ep
from megatron.lite.primitive.kernels.sgl_fused_moe import COMBINE_MODES, fused_experts_bf16
from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate
from megatron.lite.primitive.kernels.sgl_swiglu import swigluoai
from megatron.lite.primitive.kernels.visible import visible_forward

SHARED_MODES = ("separate", "fused")
EXPERTS_PATHS = ("triton", "deepgemm_ep")


class SglM3MoELayer(M3MoELayer):
    align_moe_shared: str = "separate"
    align_moe_combine: str = "slot"
    align_moe_ep_size: int = 1
    align_moe_experts: str = "triton"
    _routing = None  # (scores fp32 [T, K], ids [T, K]) of the last router call (SGLang slot order with the router switch)
    _logits = None  # fp32 router logits of the last call (fused shared-expert slot needs the K+1 routing)

    @classmethod
    def swap_in(cls, module: M3MoELayer, *, shared: str = "separate", combine: str = "slot", ep_size: int = 1,
                experts: str = "triton") -> "SglM3MoELayer":
        from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

        if shared not in SHARED_MODES:
            raise ValueError(f"align experts shared={shared!r}; known: {SHARED_MODES}")
        if combine not in COMBINE_MODES:
            raise ValueError(f"align experts combine={combine!r}; known: {COMBINE_MODES}")
        if experts not in EXPERTS_PATHS:
            raise ValueError(f"align experts experts={experts!r}; known: {EXPERTS_PATHS}")
        if (combine == "ep_normal" or experts == "deepgemm_ep") and shared == "fused":
            raise ValueError("align experts: SGLang fuses the shared expert only with EP=1; use shared='separate' with the EP paths")
        if (combine == "ep_normal" or experts == "deepgemm_ep") and (ep_size < 1 or module.experts.num_local_experts % ep_size):
            raise ValueError(f"align experts: ep_size={ep_size} must divide the expert count {module.experts.num_local_experts}")
        if experts == "deepgemm_ep":
            if ep_size < 2:
                raise ValueError("align experts: deepgemm_ep is SGLang's DeepEP path, ep_size >= 2 (EP=1 runs the Triton path)")
            if combine == "expert_asc":
                raise ValueError("align experts: deepgemm_ep combines in DeepEP's rank order; combine must be 'slot' (default) or 'ep_normal'")
            combine = "ep_normal"
        if module.dispatcher.ep_size != 1 or module.experts.etp_group is not None:
            raise NotImplementedError("align experts: ep=1 only (the EP combine contract is not fixed yet)")
        if not isinstance(module.router, SglSigmoidTopKRouter):
            raise NotImplementedError("align experts needs the router switch (SGLang slot order and weight bytes)")
        module.__class__ = cls
        module.align_moe_shared, module.align_moe_combine, module.align_moe_ep_size = shared, combine, ep_size
        module.align_moe_experts = experts
        module.router.register_forward_hook(lambda m, inp, out: setattr(module, "_routing", (out[0].detach(), out[1].detach())))
        if shared == "fused":
            module.router.gate.register_forward_hook(lambda m, inp, out: setattr(module, "_logits", out.detach()))
        return module

    def _stacked_expert_weights(self, extra13=None, extra2=None) -> tuple[torch.Tensor, torch.Tensor]:
        """[E(+1), 2I, H] and [E(+1), H, I] from TE GroupedLinear's per-expert parameters (one copy; C10 item)."""
        n = self.experts.num_local_experts
        w13 = [getattr(self.experts.fc1, f"weight{e}").detach() for e in range(n)]
        w2 = [getattr(self.experts.fc2, f"weight{e}").detach() for e in range(n)]
        if extra13 is not None:
            w13.append(extra13)
            w2.append(extra2)
        return torch.stack(w13), torch.stack(w2)

    def _visible(self, x2: torch.Tensor) -> torch.Tensor:
        cfg = self.shared_expert.config
        alpha, limit = cfg.swiglu_alpha, cfg.swiglu_limit
        scores, ids = self._routing  # the router's returned routing (SGLang slot order; replay-aware)
        w13_s = self.shared_expert.gate_up.linear.weight.detach()
        w2_s = self.shared_expert.down.linear.weight.detach()
        if self.align_moe_shared == "separate":
            w13, w2 = self._stacked_expert_weights()
            if self.align_moe_experts == "deepgemm_ep":
                routed = fused_experts_deepgemm_ep(x2, w13, w2, scores, ids, alpha, limit, ep_size=self.align_moe_ep_size)
            else:
                routed = fused_experts_bf16(x2, w13, w2, scores, ids, alpha, limit, combine=self.align_moe_combine, ep_size=self.align_moe_ep_size)
            shared = linear_bf16(swigluoai(linear_bf16(x2, w13_s), alpha, limit), w2_s)
            return routed + shared
        # fused: SGLang's K+1 launch; routed slots follow the router's routing, the shared slot takes the kernel's weight
        k = ids.shape[1]
        weights_k1, _ = moe_fused_gate(
            self._logits, self.router.expert_bias, k + 1, num_fused_shared_experts=1, renormalize=True,
            routed_scaling_factor=self.router.scaling_factor or 1.0, apply_routed_scaling_factor_on_output=True,
        )
        n = self.experts.num_local_experts
        slots = torch.cat([ids.to(torch.int32), ids.new_full((ids.shape[0], 1), n, dtype=torch.int32)], dim=1)
        weights = torch.cat([scores.float(), weights_k1[:, k:]], dim=1)
        w13, w2 = self._stacked_expert_weights(w13_s, w2_s)
        return fused_experts_bf16(x2, w13, w2, weights, slots, alpha, limit, combine=self.align_moe_combine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from megatron.lite.primitive.recompute import is_recomputing

        native = super().forward(x)  # runs the router (hooks capture its outputs) and lite's dispatcher/experts graph
        if is_recomputing():  # checkpoint replay: only the native graph matters
            self._routing = self._logits = None
            return native
        with torch.no_grad():
            visible = self._visible(x.reshape(-1, x.size(-1))).view(x.shape).to(x.dtype)
        self._routing = self._logits = None
        return visible_forward(visible, native)
