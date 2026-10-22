# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SglSigmoidTopKRouter swapped onto a lite SigmoidTopKRouter: forward bytes from the SGLang path,
backward from the formula, parameters untouched."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

H, E, K = 256, 128, 4


def _router():
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter

    cfg = SimpleNamespace(num_experts_per_tok=K, n_routed_experts=E, routed_scaling_factor=2.0, scoring_func="sigmoid",
                          hidden_size=H, aux_loss_alpha=0.0, n_group=None, topk_group=None)
    ps = SimpleNamespace(tp_size=1, tp_group=None)
    torch.manual_seed(0)
    r = SigmoidTopKRouter(cfg, ps, compute_aux_loss=False, router_dtype=torch.float32, gate_fp32=True,
                          expert_bias_persistent=True).cuda().to(torch.bfloat16)
    with torch.no_grad():
        r.gate.weight.copy_(torch.randn_like(r.gate.weight) * 0.02)
        r.expert_bias.normal_(std=0.05)
    return r


@pytest.mark.parametrize("gate", ["bf16", "fp32"])
def test_forward_bytes_follow_sglang_and_params_are_kept(gate):
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    r = _router()
    names = dict(r.named_parameters()).keys()
    w_before = r.gate.weight.detach().clone()
    SglSigmoidTopKRouter.swap_in(r, gate=gate)
    assert dict(r.named_parameters()).keys() == names and torch.equal(r.gate.weight, w_before)
    assert r.gate.weight.dtype == torch.float32  # gate_fp32 master weight
    x = torch.randn(64, H, device="cuda", dtype=torch.bfloat16)
    scores, ids = r(x)
    if gate == "bf16":
        logits = torch.mm(x, r.gate.weight.to(torch.bfloat16).t(), out_dtype=torch.float32)
    else:
        logits = torch.mm(x.float(), r.gate.weight.float().t())
    kw, kid = moe_fused_gate(logits, r.expert_bias, K, routed_scaling_factor=2.0, apply_routed_scaling_factor_on_output=True)
    assert torch.equal(ids, kid.long()) and torch.equal(scores.detach(), kw)


def test_backward_matches_formula():
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    r = SglSigmoidTopKRouter.swap_in(_router(), gate="bf16")
    x = torch.randn(64, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores, ids = r(x)
    g = torch.randn_like(scores)
    scores.backward(g)

    def _ste_bf16(t):  # forward uses the bf16-rounded value, gradient passes straight through (no bf16 grad rounding)
        return t + (t.detach().to(torch.bfloat16).float() - t.detach())

    xr = x.detach().float().requires_grad_(True)
    wr = r.gate.weight.detach().clone().requires_grad_(True)
    logits = torch.mm(_ste_bf16(xr), _ste_bf16(wr).t())
    sel = torch.gather(torch.sigmoid(logits), 1, ids)
    probs = sel / sel.sum(1, keepdim=True) * 2.0
    probs.backward(g)
    torch.testing.assert_close(r.gate.weight.grad, wr.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(x.grad.float(), xr.grad, rtol=2e-2, atol=1e-4)


def test_lite_and_sgl_router_select_the_same_experts_on_real_shapes():
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    r = _router()
    x = torch.randn(2048, H, device="cuda", dtype=torch.bfloat16)
    lite_scores, lite_ids = r(x)
    SglSigmoidTopKRouter.swap_in(r, gate="fp32")
    sgl_scores, sgl_ids = r(x)
    same = (lite_ids.sort(1).values == sgl_ids.sort(1).values).all(1)
    ls = torch.gather(lite_scores, 1, lite_ids.argsort(1))
    ss = torch.gather(sgl_scores, 1, sgl_ids.argsort(1))
    print(f"\nlite vs sgl router (fp32 gate): same expert set {same.float().mean().item():.4f}, "
          f"score bitwise mismatches {(ls != ss).sum().item()}/{ls.numel()}, max abs {(ls - ss).abs().max().item():.2e}")
    assert same.float().mean().item() > 0.99
