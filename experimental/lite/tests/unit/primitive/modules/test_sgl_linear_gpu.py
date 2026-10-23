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
    visible[:8] = torch.tensor([0.0, 1.0, -1.0, 3e-38, 1e30, -1e30, 2.0, -2.0], device="cuda", dtype=torch.bfloat16)
    native_vals = native.detach().clone()
    native_vals[:8] = torch.tensor([5.0, 0.0, 1.0, -1e30, 0.0, 1e-30, 2.0 + 2 ** -7, -2.0 * (1 - 2 ** -8)], device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        native.copy_(native_vals)
    out = visible_forward(visible, native)
    assert torch.equal(out.view(torch.int16), visible.view(torch.int16))  # bitwise, including where |a - b| is large
    g = torch.randn_like(out)
    out.backward(g)
    assert torch.equal(native.grad, g)


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
