# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""visible_forward glue and the o_proj swap: forward bytes from DeepGEMM, backward bitwise TE's."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

H, S, B = 512, 96, 2


def _deepgemm():
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module

    try:
        return deep_gemm_module()
    except ImportError as exc:
        pytest.skip(str(exc))


def test_visible_forward_is_exact_and_routes_gradient_to_native():
    from megatron.lite.primitive.kernels.visible import visible_forward

    torch.manual_seed(0)
    native = torch.randn(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    visible = native.detach() * (1 + 2 ** -7)  # 1-ulp-ish perturbation
    visible[:10] = torch.tensor([0.0, 1.0, -1.0, 3e-38, 1e30, -1e30, 2.0, -2.0, -0.0, float("inf")], device="cuda", dtype=torch.bfloat16)
    native_vals = native.detach().clone()
    native_vals[:10] = torch.tensor([5.0, 0.0, 1.0, -1e30, 0.0, 1e-30, 2.0 + 2 ** -7, -2.0 * (1 - 2 ** -8), 1.0, float("nan")],
                                    device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        native.copy_(native_vals)
    out = visible_forward(visible, native)
    assert torch.equal(out.view(torch.int16), visible.view(torch.int16))  # bitwise incl. -0.0 and non-finite native
    g = torch.randn_like(out)
    out.backward(g)
    assert torch.equal(native.grad, g)
    assert visible.grad is None


def _ps():
    return SimpleNamespace(tp_size=1, tp_rank=0, tp_group=None)


def test_o_proj_swap_forward_is_deepgemm_bytes_and_backward_is_te():
    from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
    from megatron.lite.primitive.parallel.linear import RowParallelLinear
    from megatron.lite.primitive.parallel.linear_sgl import SglRowParallelLinear

    _deepgemm()
    torch.manual_seed(1)
    ref = RowParallelLinear(H, H, _ps(), bias=False).cuda()
    with torch.no_grad():
        ref.linear.weight.normal_(std=0.02)
    swapped = RowParallelLinear(H, H, _ps(), bias=False).cuda()
    with torch.no_grad():
        swapped.linear.weight.copy_(ref.linear.weight)
    SglRowParallelLinear.swap_in(swapped)
    assert dict(swapped.named_parameters()).keys() == dict(ref.named_parameters()).keys()

    x = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16)
    x_ref, x_sw = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    y_ref, y_sw = ref(x_ref), swapped(x_sw)
    want = linear_bf16(x, swapped.linear.weight)
    assert y_sw.shape == y_ref.shape and torch.equal(y_sw, want)
    n_diff = (y_sw != y_ref).sum().item()
    print(f"\nTE vs DeepGEMM o_proj forward: {n_diff}/{y_ref.numel()} bf16 mismatches")
    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y_sw.backward(g)
    assert torch.equal(x_sw.grad, x_ref.grad)  # same TE backward graph, same bytes
    assert torch.equal(swapped.linear.weight.grad, ref.linear.weight.grad)


@pytest.mark.parametrize("with_norm_out", [False, True])
def test_qkv_swap_shows_sglang_norm_and_gemm_bytes_and_keeps_te_backward(with_norm_out):
    from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
    from megatron.lite.primitive.kernels.sgl_norm import gemma_fused_add_rmsnorm_kernel, gemma_rmsnorm_kernel
    from megatron.lite.primitive.parallel.linear import ColumnParallelLinear
    from megatron.lite.primitive.parallel.linear_sgl import SglColumnParallelLinear

    _deepgemm()
    try:
        gemma_rmsnorm_kernel(), gemma_fused_add_rmsnorm_kernel()
    except ImportError as exc:
        pytest.skip(str(exc))
    torch.manual_seed(2)

    def build():
        m = ColumnParallelLinear(H, 3 * H, _ps(), bias=False, normalization="RMSNorm", eps=1e-6, zero_centered_gamma=True,
                                 return_layernorm_output=with_norm_out).cuda()
        return m

    ref, sw = build(), build()
    with torch.no_grad():
        ref.linear.weight.normal_(std=0.02)
        ref.linear.layer_norm_weight.normal_(std=0.1)
        sw.linear.weight.copy_(ref.linear.weight)
        sw.linear.layer_norm_weight.copy_(ref.linear.layer_norm_weight)
    SglColumnParallelLinear.swap_in(sw)
    delta = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16) * 3
    x = residual + delta  # the bf16 residual stream lite feeds into the layer
    w_norm = ref.linear.layer_norm_weight.detach().to(torch.bfloat16)

    # no pair (layer 0): standalone gemma_rmsnorm of x
    out_sw = sw(x)
    out_sw, norm_sw = out_sw if with_norm_out else (out_sw, None)
    normed0 = gemma_rmsnorm_kernel()(x.reshape(-1, H).contiguous(), w_norm, 1e-6).reshape(x.shape)
    assert torch.equal(out_sw, linear_bf16(normed0, sw.linear.weight))
    if with_norm_out:
        assert torch.equal(norm_sw, normed0)

    # with the previous layer's pair: SGLang fused add+norm of the fp32 sum
    sw._residual_pair = (delta, residual)
    x_ref, x_sw = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    out_ref = ref(x_ref)
    out_sw = sw(x_sw)
    sw._residual_pair = None
    out_ref, norm_ref = out_ref if with_norm_out else (out_ref, None)
    out_sw, norm_sw = out_sw if with_norm_out else (out_sw, None)
    hid, res = delta.reshape(-1, H).clone(), residual.reshape(-1, H).clone()
    gemma_fused_add_rmsnorm_kernel()(hid, res, w_norm, 1e-6)
    normed = hid.reshape(x.shape)
    assert torch.equal(out_sw, linear_bf16(normed, sw.linear.weight))
    print(f"\nqkv: fused-add-norm+DeepGEMM vs TE LayerNormLinear bf16 mismatches {(out_sw != out_ref).sum().item()}/{out_ref.numel()}")
    if with_norm_out:
        assert torch.equal(norm_sw, normed)
    g = torch.randn_like(out_ref)
    if with_norm_out:  # one backward: TE's autograd function frees its saved tensors after the first pass
        g2 = torch.randn_like(norm_ref)
        ((out_ref * g).sum() + (norm_ref * g2).sum()).backward()
        ((out_sw * g).sum() + (norm_sw * g2).sum()).backward()
    else:
        out_ref.backward(g)
        out_sw.backward(g)
    assert torch.equal(x_sw.grad, x_ref.grad)
    assert torch.equal(sw.linear.weight.grad, ref.linear.weight.grad)
    assert torch.equal(sw.linear.layer_norm_weight.grad, ref.linear.layer_norm_weight.grad)


def test_dense_mlp_swap_shows_sglang_bytes_and_keeps_te_backward():
    from megatron.lite.model.minimax_m3.align.dense_mlp import SglM3DenseMLP
    from megatron.lite.model.minimax_m3.lite.model import M3DenseMLP
    from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
    from megatron.lite.primitive.kernels.sgl_norm import gemma_fused_add_rmsnorm_kernel, visible_input_norm
    from megatron.lite.primitive.kernels.sgl_swiglu import swigluoai

    _deepgemm()
    try:
        gemma_fused_add_rmsnorm_kernel()
    except ImportError as exc:
        pytest.skip(str(exc))
    cfg = SimpleNamespace(hidden_size=H, dense_intermediate_size=1024, rms_norm_eps=1e-6, swiglu_alpha=1.702, swiglu_limit=7.0,
                          swiglu_up_offset=1.0)
    torch.manual_seed(4)
    ref, sw = M3DenseMLP(cfg, _ps()).cuda(), M3DenseMLP(cfg, _ps()).cuda()
    with torch.no_grad():
        for m in (ref.gate_up.linear, ref.down.linear):
            m.weight.normal_(std=0.02)
        ref.gate_up.linear.layer_norm_weight.normal_(std=0.1)
        sw.gate_up.linear.weight.copy_(ref.gate_up.linear.weight)
        sw.gate_up.linear.layer_norm_weight.copy_(ref.gate_up.linear.layer_norm_weight)
        sw.down.linear.weight.copy_(ref.down.linear.weight)
    SglM3DenseMLP.swap_in(sw)
    delta = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16) * 3
    x = residual + delta
    sw._residual_pair = (delta, residual)
    x_ref, x_sw = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    y_ref, y_sw = ref(x_ref), sw(x_sw)
    sw._residual_pair = None
    normed = visible_input_norm(x, sw.gate_up.linear.layer_norm_weight, 1e-6, (delta, residual))
    want = linear_bf16(swigluoai(linear_bf16(normed, sw.gate_up.linear.weight), 1.702, 7.0), sw.down.linear.weight)
    assert torch.equal(y_sw, want)
    print(f"\ndense MLP: SGLang path vs TE path bf16 mismatches {(y_sw != y_ref).sum().item()}/{y_ref.numel()}")
    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y_sw.backward(g)
    assert torch.equal(x_sw.grad, x_ref.grad)
    for a, b in zip(ref.parameters(), sw.parameters()):
        assert torch.equal(a.grad, b.grad)


def test_lm_head_swap_and_token_loss_show_sglang_bytes_and_keep_native_backward():
    from megatron.lite.model.minimax_m3.align.lm_head import SglVocabParallelOutput, sgl_token_loss
    from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
    from megatron.lite.primitive.kernels.sgl_log_softmax import token_logprob
    from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
    from megatron.lite.primitive.parallel.linear import VocabParallelOutput

    _deepgemm()
    V, T = 4096, 48
    torch.manual_seed(5)
    ref, sw = VocabParallelOutput(V, H, _ps()).cuda(), VocabParallelOutput(V, H, _ps()).cuda()
    with torch.no_grad():
        ref.col.linear.weight.normal_(std=0.02)
        sw.col.linear.weight.copy_(ref.col.linear.weight)
    SglVocabParallelOutput.swap_in(sw)
    x = torch.randn(T, 1, H, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, V, (T, 1), device="cuda")
    x_ref, x_sw = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    logits_ref, logits_sw = ref(x_ref), sw(x_sw)
    assert torch.equal(logits_sw, linear_bf16(x, sw.col.linear.weight))
    nll_ref = vocab_parallel_cross_entropy(logits_ref, labels, None)
    nll_sw = sgl_token_loss(logits_sw, labels, None)
    assert torch.equal(nll_sw, -token_logprob(logits_sw.detach().float(), labels))
    print(f"\nlm_head: DeepGEMM vs vanilla matmul logits mismatches {(logits_sw != logits_ref).sum().item()}/{logits_ref.numel()}; "
          f"BI log_softmax nll vs CE nll mismatches {(nll_sw != nll_ref).sum().item()}/{T}, max |d| {(nll_sw - nll_ref).abs().max().item():.2e}")
    g = torch.randn_like(nll_ref)
    (nll_ref * g).sum().backward()
    (nll_sw * g).sum().backward()
    assert torch.equal(x_sw.grad, x_ref.grad)
    assert torch.equal(sw.col.linear.weight.grad, ref.col.linear.weight.grad)
