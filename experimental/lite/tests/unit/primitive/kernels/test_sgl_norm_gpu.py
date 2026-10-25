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
    ref = _ref(x, norm.weight, 1e-6)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)  # it is a Gemma (1+w) RMSNorm with this eps
    assert not torch.equal(out, kernel(x.reshape(-1, shape[-1]).contiguous(), norm.weight.data, 1e-1).reshape(shape))
    print(f"\nkernel vs torch-fp32 bf16 mismatches: {(out != ref.to(torch.bfloat16)).sum().item()}/{out.numel()}")


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
    old = te.RMSNorm(512, eps=1e-6, zero_centered_gamma=True, params_dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        old.weight.normal_(std=0.1)
    new = SglGemmaRMSNorm.from_te(old, eps=1e-6)
    assert dict(new.named_parameters()).keys() == {"weight"}
    assert torch.equal(new.weight, old.weight) and new.weight.dtype == old.weight.dtype
    x = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
    te_out, sgl_out = old(x), new(x)
    assert te_out.dtype == sgl_out.dtype == torch.bfloat16, (te_out.dtype, sgl_out.dtype)
    torch.testing.assert_close(te_out.float(), sgl_out.float(), rtol=2e-2, atol=2e-2)
    diff = (te_out.float() - sgl_out.float()).abs()
    print(f"\nTE vs sgl gemma_rmsnorm: weight {old.weight.dtype}, bf16 mismatches {(te_out != sgl_out).sum().item()}/{x.numel()}, "
          f"max abs {diff.max().item():.3e}, max rel {(diff / te_out.float().abs().clamp_min(1e-6)).max().item():.3e}")


def test_fused_add_is_the_inference_kernel_and_not_the_standalone_norm_of_the_bf16_sum():
    from megatron.lite.primitive.kernels.sgl_norm import gemma_fused_add_rmsnorm, gemma_fused_add_rmsnorm_kernel

    kernel = _kernel()
    try:
        fused_kernel = gemma_fused_add_rmsnorm_kernel()
    except ImportError as exc:
        pytest.skip(str(exc))
    torch.manual_seed(2)
    a = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16) * 3
    w = (torch.randn(2048, device="cuda") * 0.1).to(torch.bfloat16)
    normed, res = gemma_fused_add_rmsnorm(a, r, w, 1e-6)
    hid, want_res = a.clone(), r.clone()
    fused_kernel(hid, want_res, w, 1e-6)
    assert torch.equal(normed, hid) and torch.equal(res, want_res)
    assert torch.equal(res, a + r)  # carried residual == torch bf16 add (one rounding of the fp32 sum)
    torch.testing.assert_close(normed.float(), _ref(a.float() + r.float(), w, 1e-6), rtol=2e-2, atol=2e-2)
    standalone = kernel(a + r, w, 1e-6)  # norm of the bf16-rounded sum: NOT what inference computes
    n_diff = (standalone != normed).sum().item()
    print(f"\nfused-add vs standalone-on-bf16-sum mismatches: {n_diff}/{normed.numel()}")
    assert n_diff > 0


def test_fused_add_backward_matches_fp32_reference():
    from megatron.lite.primitive.kernels.sgl_norm import gemma_fused_add_rmsnorm, gemma_fused_add_rmsnorm_kernel

    try:
        gemma_fused_add_rmsnorm_kernel()
    except ImportError as exc:
        pytest.skip(str(exc))
    torch.manual_seed(3)
    a = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    r = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.nn.Parameter((torch.randn(1024, device="cuda") * 0.1).to(torch.bfloat16))
    g1 = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    g2 = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    normed, res = gemma_fused_add_rmsnorm(a, r, w, 1e-6)
    (normed * g1).sum().backward(retain_graph=True)
    (res * g2).sum().backward()

    ar, rr, wr = (t.detach().float().requires_grad_(True) for t in (a, r, w))
    y = ar + rr
    (_ref(y, wr, 1e-6) * g1.float()).sum().backward(retain_graph=True)
    (y * g2.float()).sum().backward()
    for got, ref in ((a.grad, ar.grad), (r.grad, rr.grad)):
        torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(w.grad.float(), wr.grad, rtol=2e-2, atol=2e-1)


def _fork_triton_norms():
    import importlib.util, os

    root = os.environ.get("SGLANG_FORK_PY")
    path = os.path.join(root or "", "kernels", "ops", "layernorm", "minimax_m3_rmsnorm.py")
    if not root or not os.path.exists(path):
        pytest.skip("SGLANG_FORK_PY (fork python/sglang dir) with kernels/ops/layernorm/minimax_m3_rmsnorm.py not available")
    spec = importlib.util.spec_from_file_location("fork_minimax_m3_rmsnorm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("shape", [(4096, 6144), (16384, 128), (7, 3, 6144)])
def test_contract_norms_match_the_fork_triton_source_bitwise(shape):
    """The vendored kernels and the fork's minimax_m3_rmsnorm (what the server runs under deterministic inference)."""
    from megatron.lite.primitive.kernels.sgl_norm import gemma_fused_add_rmsnorm_kernel, gemma_rmsnorm_kernel

    fork = _fork_triton_norms()
    torch.manual_seed(3)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 2
    r = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 2
    w = (torch.rand(shape[-1], device="cuda") - 0.5).to(torch.bfloat16)
    assert torch.equal(gemma_rmsnorm_kernel()(x, w, 1e-6), fork.gemma_rmsnorm(x, w, 1e-6))
    hid, res = x.clone(), r.clone()
    gemma_fused_add_rmsnorm_kernel()(hid, res, w, 1e-6)
    out, summed = fork.gemma_fused_add_rmsnorm(x, r, w, 1e-6)
    assert torch.equal(hid, out) and torch.equal(res, summed)


def test_contract_fused_add_tracks_the_compiled_inference_kernel_to_rare_ulp_events():
    """Same math as the compiled sgl_kernel/flashinfer kernel; only rare 1-ulp differences (toolchain-dependent)."""
    from megatron.lite.primitive.kernels.sgl_norm import gemma_fused_add_rmsnorm_inference_kernel, gemma_fused_add_rmsnorm_kernel

    try:
        aot = gemma_fused_add_rmsnorm_inference_kernel()
    except ImportError as exc:
        pytest.skip(str(exc))
    torch.manual_seed(4)
    x = torch.randn(4096, 6144, device="cuda", dtype=torch.bfloat16) * 2
    r = torch.randn(4096, 6144, device="cuda", dtype=torch.bfloat16) * 2
    w = (torch.rand(6144, device="cuda") - 0.5).to(torch.bfloat16)
    h1, r1 = x.clone(), r.clone(); gemma_fused_add_rmsnorm_kernel()(h1, r1, w, 1e-6)
    h2, r2 = x.clone(), r.clone(); aot(h2, r2, w, 1e-6)
    assert torch.equal(r1, r2)
    frac = (h1 != h2).float().mean().item()
    print(f"\ncontract vs compiled fused-add norm: {frac * 100:.5f}% elements differ")
    assert frac < 1e-4
    torch.testing.assert_close(h1.float(), h2.float(), rtol=1e-2, atol=1e-2)
