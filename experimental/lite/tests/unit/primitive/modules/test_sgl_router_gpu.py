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


def _needs_deepgemm(gate):
    if gate == "fp32_deepgemm":
        from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module

        try:
            deep_gemm_module()
        except ImportError as exc:
            pytest.skip(str(exc))


@pytest.mark.parametrize("gate", ["bf16", "fp32", "fp32_deepgemm"])
def test_forward_bytes_follow_sglang_and_params_are_kept(gate):
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    _needs_deepgemm(gate)
    r = _router()
    names = dict(r.named_parameters()).keys()
    w_before = r.gate.weight.detach().clone()
    SglSigmoidTopKRouter.swap_in(r, gate=gate)
    assert dict(r.named_parameters()).keys() == names and torch.equal(r.gate.weight, w_before)
    assert r.gate.weight.dtype == torch.float32  # gate_fp32 master weight
    hook_calls = []
    r.gate.register_forward_pre_hook(lambda m, inp: hook_calls.append(1))  # the DDP param-gather hook sits here
    x = torch.randn(64, H, device="cuda", dtype=torch.bfloat16)
    scores, ids = r(x)
    assert hook_calls == [1]
    if gate == "bf16":
        logits = torch.mm(x, r.gate.weight.to(torch.bfloat16).t(), out_dtype=torch.float32)
    elif gate == "fp32":
        logits = torch.mm(x.float(), r.gate.weight.float().t())
    else:
        from megatron.lite.primitive.kernels.sgl_deepgemm import bf16xfp32_gemm

        logits = bf16xfp32_gemm(x, r.gate.weight)
    kw, kid = moe_fused_gate(logits, r.expert_bias, K, routed_scaling_factor=2.0, apply_routed_scaling_factor_on_output=True)
    assert torch.equal(ids, kid.long()) and torch.equal(scores.detach(), kw)


@pytest.mark.parametrize("gate", ["bf16", "fp32", "fp32_deepgemm"])
def test_backward_matches_formula(gate):
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    _needs_deepgemm(gate)
    r = SglSigmoidTopKRouter.swap_in(_router(), gate=gate)
    x = torch.randn(64, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    logits_seen = []
    r.gate.register_forward_hook(lambda m, inp, out: logits_seen.append(out.detach()))
    scores, ids = r(x)
    g = torch.randn_like(scores)
    scores.backward(g)

    def _ste_bf16(t):  # forward uses the bf16-rounded value, gradient passes straight through (no bf16 grad rounding)
        return t + (t.detach().to(torch.bfloat16).float() - t.detach())

    xr = x.detach().float().requires_grad_(True)
    wr = r.gate.weight.detach().clone().requires_grad_(True)
    if gate == "fp32_deepgemm":  # forward operand is the bf16 high/low split of W
        from megatron.lite.primitive.kernels.sgl_deepgemm import effective_weight

        w_fwd = wr + (effective_weight(wr) - wr.detach())
    else:
        w_fwd = _ste_bf16(wr) if gate == "bf16" else wr
    logits = torch.mm(_ste_bf16(xr), w_fwd.t())
    logits = logits + (logits_seen[0] - logits.detach())  # the module's own logit values, formula gradient
    sel = torch.gather(torch.sigmoid(logits), 1, ids)
    probs = sel / sel.sum(1, keepdim=True) * 2.0
    probs.backward(g)
    torch.testing.assert_close(r.gate.weight.grad, wr.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(x.grad.float(), xr.grad, rtol=5e-3, atol=1e-5)  # dX is bf16-rounded, else exact formula


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
    # rows with a clear 4th-vs-5th biased-score gap must agree exactly; near-ties may legitimately differ
    biased = torch.sigmoid(torch.mm(x.float(), r.gate.weight.float().t())) + r.expert_bias
    top5 = biased.topk(5, dim=1).values
    clear = (top5[:, 3] - top5[:, 4]) > 1e-5
    print(f"\nlite vs sgl router (fp32 gate): same expert set {same.float().mean().item():.4f} "
          f"({int(clear.sum())}/{x.size(0)} clear rows), score bitwise mismatches {(ls != ss).sum().item()}/{ls.numel()}, "
          f"max abs {(ls - ss).abs().max().item():.2e}")
    assert bool(same[clear].all())
    torch.testing.assert_close(ss[same], ls[same], rtol=1e-5, atol=1e-6)


_CUBLAS_NOT_BI = pytest.mark.xfail(
    strict=True,
    reason="cuBLAS torch.mm is not batch-invariant (GB200 10-22: M=1..1024 vs 4096 differ in ~99% of logits, "
    "same M reproducible); the contract mode is fp32_deepgemm",
)


@pytest.mark.parametrize(
    "gate", [pytest.param("bf16", marks=_CUBLAS_NOT_BI), pytest.param("fp32", marks=_CUBLAS_NOT_BI), "fp32_deepgemm"]
)
def test_gate_gemm_is_batch_invariant_at_model_width(gate):
    """The gate GEMM is the only non-vendored stage: its logits must not depend on how many rows share the launch."""
    from megatron.lite.primitive.modules.router_sgl import _GateMM

    _needs_deepgemm(gate)
    torch.manual_seed(5)
    w = (torch.randn(E, 6144, device="cuda") * 0.02).requires_grad_(False)
    x = torch.randn(4096, 6144, device="cuda", dtype=torch.bfloat16)
    full = _GateMM.apply(x, w, gate)
    report = {n: (_GateMM.apply(x[:n], w, gate) != full[:n]).sum().item() for n in (1, 8, 37, 512, 2048)}
    print(f"\ngate {gate}: logits mismatches vs the 4096-row launch {report}")
    assert all(v == 0 for v in report.values()), report


def test_fused_shared_expert_launch_gives_the_same_routed_bytes():
    """SGLang (EP=1) launches moe_fused_gate with K=5 / K_ROUTED=4 (fused shared expert); align launches K=4."""
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate

    torch.manual_seed(6)
    logits = torch.randn(4096, E, device="cuda")
    bias = torch.randn(E, device="cuda") * 0.05
    w4, i4 = moe_fused_gate(logits, bias, K, routed_scaling_factor=2.0, apply_routed_scaling_factor_on_output=True)
    w5, i5 = moe_fused_gate(logits, bias, K + 1, num_fused_shared_experts=1, routed_scaling_factor=2.0,
                            apply_routed_scaling_factor_on_output=True)
    assert torch.equal(i4, i5[:, :K]) and torch.equal(w4, w5[:, :K])
    assert torch.equal(i5[:, K], torch.full((4096,), E, dtype=torch.int32, device="cuda"))


def test_replay_keeps_kernel_bytes_on_unchanged_rows():
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    r = SglSigmoidTopKRouter.swap_in(_router(), gate="fp32")
    x = torch.randn(32, H, device="cuda", dtype=torch.bfloat16)
    scores0, ids0 = r(x)
    override = ids0.clone()
    override[0] = torch.tensor([0, 1, 2, 3], device="cuda")

    class FakeReplay:  # RouterReplay.select_indices contract: returns a fresh tensor
        def select_indices(self, native):
            return override.clone()

    r.router_replay = FakeReplay()
    scores, ids = r(x)
    assert torch.equal(ids, override)
    assert torch.equal(scores[1:], scores0[1:])
    assert not torch.equal(ids[0], ids0[0]) and torch.isfinite(scores[0]).all()
