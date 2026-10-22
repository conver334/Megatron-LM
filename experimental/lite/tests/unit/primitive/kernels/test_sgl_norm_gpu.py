# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SglGemmaRMSNorm: forward bytes == the inference kernel, backward == fp32 reference gradient."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]


def _kernel():
    from megatron.lite.primitive.kernels.sgl_norm import gemma_rmsnorm_kernel

    try:
        return gemma_rmsnorm_kernel()
    except ImportError as exc:
        pytest.skip(str(exc))


def _ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())


@pytest.mark.parametrize("shape", [(64, 512), (7, 3, 6144)])
def test_forward_is_the_inference_kernel_bytes(shape):
    from megatron.lite.primitive.kernels.sgl_norm import SglGemmaRMSNorm

    kernel = _kernel()
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    norm = SglGemmaRMSNorm(shape[-1], 1e-6, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_(std=0.1)
    out = norm(x)
    want = kernel(x.reshape(-1, shape[-1]).contiguous(), norm.weight.data, 1e-6).reshape(shape)
    assert out.dtype == torch.bfloat16 and torch.equal(out, want)
    # the kernel is not the torch fp32 rounding; record how far (information, not a gate)
    ref = _ref(x, norm.weight, 1e-6).to(torch.bfloat16)
    print(f"\nkernel vs torch-fp32 bf16 mismatches: {(out != ref).sum().item()}/{out.numel()}")


def test_backward_matches_fp32_reference_gradient():
    from megatron.lite.primitive.kernels.sgl_norm import SglGemmaRMSNorm

    _kernel()
    torch.manual_seed(1)
    x = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    norm = SglGemmaRMSNorm(1024, 1e-6, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_(std=0.1)
    g = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
    norm(x).backward(g)

    xr = x.detach().float().requires_grad_(True)
    wr = norm.weight.detach().float().requires_grad_(True)
    _ref(xr, wr, 1e-6).backward(g.float())
    torch.testing.assert_close(x.grad.float(), xr.grad, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(norm.weight.grad.float(), wr.grad, rtol=2e-2, atol=1e-1)


def test_from_te_keeps_weight_name_and_values():
    from megatron.lite.primitive import transformer_engine as te
    from megatron.lite.primitive.kernels.sgl_norm import SglGemmaRMSNorm

    _kernel()
    old = te.RMSNorm(512, eps=1e-6, zero_centered_gamma=True).cuda()
    with torch.no_grad():
        old.weight.normal_(std=0.1)
    new = SglGemmaRMSNorm.from_te(old, eps=1e-6)
    assert dict(new.named_parameters()).keys() == {"weight"}
    assert torch.equal(new.weight, old.weight) and new.weight.dtype == old.weight.dtype
    x = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
    te_out, sgl_out = old(x), new(x)
    assert te_out.dtype == sgl_out.dtype == torch.bfloat16, (te_out.dtype, sgl_out.dtype)
    diff = (te_out.float() - sgl_out.float()).abs()
    print(f"\nTE vs sgl gemma_rmsnorm: weight {old.weight.dtype}, bf16 mismatches {(te_out != sgl_out).sum().item()}/{x.numel()}, "
          f"max abs {diff.max().item():.3e}, max rel {(diff / te_out.float().abs().clamp_min(1e-6)).max().item():.3e}")
