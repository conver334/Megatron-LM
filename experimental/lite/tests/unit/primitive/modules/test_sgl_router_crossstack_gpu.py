# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cross-stack router contract: the SGLang fork's *own source* (dsv4/gemm.py deep_gemm branch, moe_fused_gate.py,
deep_gemm_wrapper/entrypoint.py) loaded by path with the sglang infrastructure stubbed, against mlite's
``fp32_deepgemm`` router, byte for byte. Needs SGLANG_FORK_PY=<fork>/python/sglang and sgl-deep-gemm;
M3_TRUNC6_CKPT=<HF dir> uses the real gate weights/bias of layers 3-5."""

from __future__ import annotations

import functools
import glob
import json
import os

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

E, H, K, RSF = 128, 6144, 4, 2.0
ROWS = (1, 8, 37, 512, 4096)


@functools.lru_cache(maxsize=None)
def _fork():
    """(linear_bf16_fp32, moe_fused_gate, is_arch_support_pdl()) from the fork's own source files."""
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module
    from sglang_fork import install_stubs, load

    try:
        deep_gemm_module()
    except ImportError as exc:
        pytest.skip(str(exc))
    pdl = install_stubs()
    gemm = load("kernels/ops/attention/dsv4/gemm.py", "sglang.kernels.ops.attention.dsv4.gemm")
    gate = load("kernels/ops/moe/moe_fused_gate.py", "sglang.kernels.ops.moe.moe_fused_gate")
    assert gemm._linear_bf16_fp32_algo == "deep_gemm"
    return gemm.linear_bf16_fp32, gate.moe_fused_gate, pdl


def _gate_params(layer: int):
    """Real trunc6 router gate (fp32 on disk) and bias when M3_TRUNC6_CKPT is set; else random with the same dtypes."""
    ckpt = os.environ.get("M3_TRUNC6_CKPT")
    if ckpt and glob.glob(f"{ckpt}/*.index.json"):
        from safetensors import safe_open

        idx = json.load(open(glob.glob(f"{ckpt}/*.index.json")[0]))["weight_map"]

        def load(name):
            with safe_open(f"{ckpt}/{idx[name]}", "pt", device="cuda") as f:
                return f.get_tensor(name)

        w = load(f"model.layers.{layer}.mlp.gate.weight")
        b = load(f"model.layers.{layer}.mlp.gate.e_score_correction_bias").float()
        assert w.dtype == torch.float32 and tuple(w.shape) == (E, H)
        return w, b, "trunc6"
    torch.manual_seed(layer)
    return torch.randn(E, H, device="cuda") * 0.02, torch.randn(E, device="cuda") * 0.05, "random"


@pytest.mark.parametrize("layer", [3, 4, 5])
def test_fork_router_path_equals_mlite_fp32_deepgemm_bytes(layer):
    from megatron.lite.primitive.kernels.sgl_deepgemm import bf16xfp32_gemm
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate as ours_gate

    fork_linear, fork_gate, pdl = _fork()
    w, bias, src = _gate_params(layer)
    torch.manual_seed(100 + layer)
    x_all = torch.randn(max(ROWS), H, device="cuda", dtype=torch.bfloat16)
    report = []
    for m in ROWS:
        x = x_all[:m]
        # 1. router logits: fork linear_bf16_fp32 (algo deep_gemm, fp32 weight) vs mlite split GEMM
        lf = fork_linear(x, w)
        lm = bf16xfp32_gemm(x, w)
        assert lf.dtype == torch.float32 and torch.equal(lf, lm), (m, (lf != lm).sum().item())
        # 2. selection + weights: fork Triton moe_fused_gate (its own PDL choice) vs vendored launcher
        wf, idf = fork_gate(lf, bias, K, renormalize=True, routed_scaling_factor=RSF, apply_routed_scaling_factor_on_output=True)
        wm, idm = ours_gate(lm, bias, K, renormalize=True, routed_scaling_factor=RSF, apply_routed_scaling_factor_on_output=True, use_pdl=pdl)
        assert torch.equal(idf, idm) and torch.equal(wf, wm), m
        wm0, idm0 = ours_gate(lm, bias, K, renormalize=True, routed_scaling_factor=RSF, apply_routed_scaling_factor_on_output=True, use_pdl=False)
        pdl_same = torch.equal(idf, idm0) and torch.equal(wf, wm0)
        # 3. SGLang EP=1 default fuses the shared expert as slot 5: routed bytes must not change
        w5, id5 = fork_gate(lf, bias, K + 1, num_fused_shared_experts=1, renormalize=True, routed_scaling_factor=RSF,
                            apply_routed_scaling_factor_on_output=True)
        assert torch.equal(id5[:, :K], idf) and torch.equal(w5[:, :K], wf), m
        report.append(f"M={m}: logits/ids/weights bitwise; PDL-off launcher identical={pdl_same}")
    print(f"\nlayer {layer} ({src} gate, fork PDL={pdl}): " + "; ".join(report), flush=True)


def test_fork_router_path_equals_mlite_router_module(monkeypatch):
    """End to end on the module: SglSigmoidTopKRouter(fp32_deepgemm) output == fork linear_bf16_fp32 + moe_fused_gate."""
    from types import SimpleNamespace

    from megatron.lite.primitive.modules.router import SigmoidTopKRouter
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    fork_linear, fork_gate, _ = _fork()
    w, bias, src = _gate_params(3)
    cfg = SimpleNamespace(num_experts_per_tok=K, n_routed_experts=E, routed_scaling_factor=RSF, scoring_func="sigmoid",
                          hidden_size=H, aux_loss_alpha=0.0, n_group=None, topk_group=None)
    r = SigmoidTopKRouter(cfg, SimpleNamespace(tp_size=1, tp_group=None), compute_aux_loss=False, router_dtype=torch.float32,
                          gate_fp32=True, expert_bias_persistent=True).cuda().to(torch.bfloat16)
    with torch.no_grad():
        r.gate.weight.copy_(w)
        r.expert_bias.copy_(bias)
    SglSigmoidTopKRouter.swap_in(r, gate="fp32_deepgemm")
    torch.manual_seed(7)
    for m in ROWS:
        x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16)
        scores, ids = r(x)
        wf, idf = fork_gate(fork_linear(x, w), bias, K, renormalize=True, routed_scaling_factor=RSF, apply_routed_scaling_factor_on_output=True)
        assert torch.equal(ids, idf.long()) and torch.equal(scores.detach(), wf), m
    print(f"\nrouter module vs fork path ({src} gate): bitwise for M in {ROWS}", flush=True)


def test_crossstack_comparison_detects_planted_errors(monkeypatch):
    """Planted errors must break byte equality: dropping the low part, or a non-power-of-two split scale.
    (A power-of-two scale change such as 1/128 is byte-neutral: the bf16 low part scales exactly.)"""
    from megatron.lite.primitive.kernels import sgl_deepgemm

    fork_linear, _, _ = _fork()
    w, bias, _ = _gate_params(3)
    torch.manual_seed(11)
    x = torch.randn(512, H, device="cuda", dtype=torch.bfloat16)
    ref = fork_linear(x, w)
    assert torch.equal(ref, sgl_deepgemm.bf16xfp32_gemm(x, w))
    hi_only = sgl_deepgemm.bf16_gemm_f32(x, sgl_deepgemm.split_fp32_weight(w)[0])
    assert not torch.equal(ref, hi_only)
    monkeypatch.setattr(sgl_deepgemm, "WEIGHT_SPLIT_SCALE", 1.0 / 128.0)
    assert torch.equal(ref, sgl_deepgemm.bf16xfp32_gemm(x, w))  # exact power-of-two rescale
    monkeypatch.setattr(sgl_deepgemm, "WEIGHT_SPLIT_SCALE", 1.0 / 3.0)
    assert not torch.equal(ref, sgl_deepgemm.bf16xfp32_gemm(x, w))
