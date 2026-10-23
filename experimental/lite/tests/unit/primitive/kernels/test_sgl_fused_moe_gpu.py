# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Vendored SGLang fused-MoE path: kernel text == fork, kernel launch == fork's launcher by path, the whole layer
== the fork's fused_experts_impl by path, torch re-implementations of moe_align_block_size / moe_sum_reduce behave
as the CUDA ops, combine-order contract modes, fp64 sanity."""

from __future__ import annotations

import inspect
import re

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

T, H, I, E, K = 96, 512, 256, 16, 4
ALPHA, LIMIT = 1.702, 7.0


def _inputs(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    w13 = (torch.randn(E, 2 * I, H, device="cuda") * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda") * 0.02).to(torch.bfloat16)
    logits = torch.randn(T, E, device="cuda")
    bias = torch.randn(E, device="cuda") * 0.05
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate

    weights, ids = moe_fused_gate(logits, bias, K, routed_scaling_factor=2.0, apply_routed_scaling_factor_on_output=True)
    return x, w13, w2, weights, ids


def test_moe_align_block_size_layout():
    from megatron.lite.primitive.kernels.sgl_fused_moe import moe_align_block_size

    topk_ids = torch.tensor([[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], device="cuda", dtype=torch.int32) - 1  # docstring example (0-based)
    sorted_ids, expert_ids, n_post = moe_align_block_size(topk_ids, 4, 4)
    assert n_post.item() == 16 and sorted_ids.shape[0] == 16 and expert_ids.tolist() == [0, 1, 2, 3]
    numel = topk_ids.numel()
    for e in range(4):
        blk = sorted_ids[4 * e : 4 * e + 4].tolist()
        real = [i for i in blk if i != numel]
        assert all(topk_ids.reshape(-1)[i].item() == e for i in real) and len(real) == 3
    assert sorted(i for i in sorted_ids.tolist() if i != numel) == list(range(numel))


def test_fused_experts_matches_fp64_semantics_and_is_deterministic():
    from megatron.lite.primitive.kernels.sgl_fused_moe import fused_experts_bf16

    x, w13, w2, weights, ids = _inputs()
    out = fused_experts_bf16(x, w13, w2, weights, ids, ALPHA, LIMIT)
    assert torch.equal(out, fused_experts_bf16(x, w13, w2, weights, ids, ALPHA, LIMIT))
    ref = torch.zeros(T, H, device="cuda", dtype=torch.float64)
    for k in range(K):
        e = ids[:, k].long()
        gu = torch.einsum("th,tnh->tn", x.double(), w13.double()[e])
        gate, up = gu.chunk(2, dim=-1)
        act = gate.clamp(max=LIMIT) * torch.sigmoid(gate.clamp(max=LIMIT) * ALPHA) * (up.clamp(-LIMIT, LIMIT) + 1)
        ref += torch.einsum("ti,thi->th", act, w2.double()[e]) * weights[:, k].double()[:, None]
    torch.testing.assert_close(out.double(), ref, rtol=3e-2, atol=3e-2)


def test_vendored_kernel_text_equals_fork_and_launch_equals_fork_launcher():
    from megatron.lite.primitive.kernels import sgl_fused_moe as V
    from megatron.lite.primitive.kernels.sgl_fused_moe import invoke_fused_moe_kernel_bf16, moe_align_block_size
    from sglang_fork import fork_root, install_stubs, load

    install_stubs()
    fork = load("kernels/ops/moe/fused_moe_triton_kernels.py", "sglang.kernels.ops.moe.fused_moe_triton_kernels")
    import ast
    import textwrap

    for name in ("fused_moe_kernel", "write_zeros_to_output"):
        ours = inspect.getsource(getattr(V, name).fn) if hasattr(getattr(V, name), "fn") else inspect.getsource(getattr(V, name))
        theirs = inspect.getsource(getattr(fork, name).fn) if hasattr(getattr(fork, name), "fn") else inspect.getsource(getattr(fork, name))
        assert ast.dump(ast.parse(textwrap.dedent(ours))) == ast.dump(ast.parse(textwrap.dedent(theirs))), name  # structure, not text
    x, w13, w2, weights, ids = _inputs(1)
    import triton.language as tl

    sorted_ids, expert_ids, n_post = moe_align_block_size(ids, 64, E)
    ours = torch.empty(T * K, 2 * I, dtype=torch.bfloat16, device="cuda")
    theirs = torch.empty_like(ours)
    invoke_fused_moe_kernel_bf16(x, w13, ours, weights, ids, sorted_ids, expert_ids, n_post, False, K)
    fork.invoke_fused_moe_kernel(  # the gate-up launch as SGLang's _fused_moe_kernel_sequence issues it (router_topk default)
        x, w13, None, theirs, None, None, None, weights, ids, sorted_ids, expert_ids, n_post, False, K,
        dict(V.DETERMINISTIC_CONFIG), compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False, block_shape=None, filter_expert=(E != E),
    )
    assert torch.equal(ours, theirs)
    act = V.swigluoai(ours, ALPHA, LIMIT)
    ours3 = torch.empty(T, K, H, dtype=torch.bfloat16, device="cuda")
    theirs3 = torch.empty_like(ours3)
    invoke_fused_moe_kernel_bf16(act, w2, ours3, weights, ids, sorted_ids, expert_ids, n_post, True, 1, router_topk=K)
    fork.invoke_fused_moe_kernel(
        act, w2, None, theirs3, None, None, None, weights, ids, sorted_ids, expert_ids, n_post, True, 1,
        dict(V.DETERMINISTIC_CONFIG), compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False,
        use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False, block_shape=None, filter_expert=False, router_topk=K,
    )
    assert torch.equal(ours3, theirs3)


def test_moe_sum_reduce_is_sequential_fp32():
    from megatron.lite.primitive.kernels.sgl_fused_moe import moe_sum_reduce

    torch.manual_seed(3)
    c3 = torch.randn(T, K, H, device="cuda", dtype=torch.bfloat16)
    out = moe_sum_reduce(c3)
    seq = ((c3[:, 0].float() + c3[:, 1].float()) + c3[:, 2].float()) + c3[:, 3].float()
    assert torch.equal(out, seq.to(torch.bfloat16))
    print(f"\nmoe_sum_reduce vs torch.sum(dim=1): {(out != c3.float().sum(1).to(torch.bfloat16)).sum().item()}/{out.numel()} bf16 mismatches")


def test_deterministic_config_matches_fork_default():
    from megatron.lite.primitive.kernels.sgl_fused_moe import DETERMINISTIC_CONFIG
    from sglang_fork import install_stubs, load

    install_stubs()
    cfg_mod = load("srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py", "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config")
    got = cfg_mod.get_default_config(4096, 128, 2 * 3072, 6144, 4, None, False)
    assert got == DETERMINISTIC_CONFIG, got
    assert cfg_mod.get_moe_configs(128, 2 * 3072, None) is None  # deterministic: no tuned (TMA) configs


def test_moe_align_block_size_multi_block_and_empty_experts():
    from megatron.lite.primitive.kernels.sgl_fused_moe import moe_align_block_size

    counts = [0, 5, 1, 70]
    flat = torch.cat([torch.full((c,), e, dtype=torch.int32) for e, c in enumerate(counts)])
    topk_ids = flat[torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))].view(-1, 4).cuda()
    for dtype in (torch.int32, torch.int64):
        sorted_ids, expert_ids, n_post = moe_align_block_size(topk_ids.to(dtype), 64, 4)
        numel = topk_ids.numel()
        assert n_post.item() == 64 + 64 + 128 and expert_ids.tolist() == [1, 2, 3, 3]
        assert (sorted_ids[5:64] == numel).all() and (sorted_ids[:5] != numel).all()
        assert (topk_ids.reshape(-1)[sorted_ids[:5].long()] == 1).all()
        assert (sorted_ids[64:65] != numel).all() and (sorted_ids[65:128] == numel).all()
        assert (sorted_ids[128:198] != numel).all() and (sorted_ids[198:256] == numel).all()
        assert (topk_ids.reshape(-1)[sorted_ids[128:198].long()] == 3).all()


def test_moe_sum_reduce_order_is_observable_with_cancellation():
    from megatron.lite.primitive.kernels.sgl_fused_moe import moe_sum_reduce

    big = 2.0 ** 25
    x = torch.tensor([[[big], [1.0], [-big], [1.0]], [[1.0], [big], [1.0], [-big]], [[big], [1.0], [-big], [0.0]]],
                     device="cuda", dtype=torch.bfloat16)  # [3, K=4, H=1]
    out = moe_sum_reduce(x)
    # fp32 absorbs the 1.0 next to 2**25: sequential ((big+1)-big)+1 = 1, ((1+big)+1)-big = 0, ((big+1)-big)+0 = 0
    assert out[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert moe_sum_reduce(x, scale=3.0)[0, 0].item() == 3.0  # scale after the fp32 sum
    pairwise = ((x[:, 0].float() + x[:, 1].float()) + (x[:, 2].float() + x[:, 3].float())).to(torch.bfloat16)
    assert not torch.equal(out, pairwise)


def _m3_fork_kwargs():
    # MiniMaxM3MoE -> FusedMoE(activation="silu", is_gated=True, gemm1_alpha, gemm1_clamp_limit, gate_up_interleaved=False),
    # routed_scaling_factor folded into the top-k weights, EP=1 -> filter_expert False
    return dict(inplace=False, activation="silu", is_gated=True, gemm1_alpha=ALPHA, gemm1_limit=LIMIT, swiglu_limit=None,
                gate_up_interleaved=False, apply_router_weight_on_input=False, routed_scaling_factor=None, filter_expert=False)


@pytest.mark.parametrize("shape", ["unit", "model_width"])
def test_whole_layer_equals_fork_fused_experts_impl(shape):
    """mlite fused_experts_bf16 == the fork's fused_experts_impl (its own orchestration, config resolution, caches,
    swiglu dispatch and Triton sum) run by path on the same input: the EP=1 MoE contract, no tolerance."""
    from megatron.lite.primitive.kernels.sgl_fused_moe import fused_experts_bf16
    from sglang_fork import load_fork_fused_moe

    fm = load_fork_fused_moe()
    if shape == "unit":
        x, w13, w2, weights, ids = _inputs(5)
    else:  # real hidden / intermediate width, a slice of the experts
        from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate

        torch.manual_seed(7)
        t, h, i_, e = 2048, 6144, 3072, 32
        x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
        w13 = (torch.randn(e, 2 * i_, h, device="cuda", dtype=torch.bfloat16) * 0.02)
        w2 = (torch.randn(e, h, i_, device="cuda", dtype=torch.bfloat16) * 0.02)
        weights, ids = moe_fused_gate(torch.randn(t, e, device="cuda"), torch.randn(e, device="cuda") * 0.05, K,
                                      routed_scaling_factor=2.0, apply_routed_scaling_factor_on_output=True)
    ours = fused_experts_bf16(x, w13, w2, weights, ids, ALPHA, LIMIT)
    theirs = fm.fused_experts_impl(x, w13, w2, weights, ids, **_m3_fork_kwargs())
    assert torch.equal(ours, theirs), f"{(ours != theirs).float().mean().item():.4f} of outputs differ"
    assert ours.abs().max().item() > 0


def test_gemm_bytes_do_not_depend_on_token_order_within_expert_blocks():
    """moe_align_block_size is layout only: permuting the token order inside each expert's run leaves every
    kernel output byte unchanged, so the torch layout stands in for the CUDA/JIT kernels' order."""
    from megatron.lite.primitive.kernels.sgl_fused_moe import DETERMINISTIC_CONFIG, invoke_fused_moe_kernel_bf16, moe_align_block_size

    x, w13, w2, weights, ids = _inputs(8)
    sorted_ids, expert_ids, n_post = moe_align_block_size(ids, DETERMINISTIC_CONFIG["BLOCK_SIZE_M"], E)
    numel = ids.numel()
    shuffled = sorted_ids.clone()
    g = torch.Generator(device="cuda").manual_seed(0)
    for e in range(E):  # reverse / permute the real entries of each expert's blocks, keep the padding where it is
        blk = (expert_ids == e).nonzero().flatten()
        if blk.numel() == 0:
            continue
        lo, hi = blk[0].item() * 64, (blk[-1].item() + 1) * 64
        real = (sorted_ids[lo:hi] != numel).nonzero().flatten() + lo
        shuffled[real] = sorted_ids[real][torch.randperm(real.numel(), generator=g, device="cuda")]
    assert not torch.equal(shuffled, sorted_ids)
    outs = []
    for layout in (sorted_ids, shuffled):
        c1 = torch.empty(T * K, 2 * I, dtype=torch.bfloat16, device="cuda")
        invoke_fused_moe_kernel_bf16(x, w13, c1, weights, ids, layout, expert_ids, n_post, False, K)
        outs.append(c1)
    assert torch.equal(outs[0], outs[1])


def test_moe_sum_reduce_starts_from_positive_zero():
    from megatron.lite.primitive.kernels.sgl_fused_moe import moe_sum_reduce

    x = torch.full((2, K, 3), -0.0, device="cuda", dtype=torch.bfloat16)
    assert (moe_sum_reduce(x).view(torch.int16) == 0).all()  # kernels: acc = 0.0f; acc += -0.0 -> +0.0


def test_combine_modes_reference_semantics_and_observability():
    from megatron.lite.primitive.kernels.sgl_fused_moe import COMBINE_MODES, combine_slots, moe_sum_reduce

    torch.manual_seed(11)
    c3 = torch.randn(T, K, H, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(T)]).to(torch.int32)
    slot = combine_slots(c3, ids, "slot")
    assert torch.equal(slot, moe_sum_reduce(c3))
    # expert_asc == per-token sort by expert id, then the same sequential sum
    ref = torch.empty_like(slot)
    for t in range(T):
        order = sorted(range(K), key=lambda k: ids[t, k].item())
        acc = torch.zeros(H, device="cuda")
        for k in order:
            acc = acc + c3[t, k].float()
        ref[t] = acc.to(torch.bfloat16)
    assert torch.equal(combine_slots(c3, ids, "expert_asc"), ref)
    # ep_normal == per-rank slot-order partials (bf16, zeros for remote slots), ranks with a slot added in rank order
    ep = 4
    ref = torch.empty_like(slot)
    for t in range(T):
        acc = torch.zeros(H, device="cuda")
        for r in range(ep):
            on_r = [k for k in range(K) if ids[t, k].item() // (E // ep) == r]
            if not on_r:
                continue
            part = torch.zeros(H, device="cuda")
            for k in range(K):
                part = part + (c3[t, k].float() if k in on_r else torch.zeros(H, device="cuda"))
            acc = acc + part.to(torch.bfloat16).float()
        ref[t] = acc.to(torch.bfloat16)
    assert torch.equal(combine_slots(c3, ids, "ep_normal", num_experts=E, ep_size=ep), ref)
    assert torch.equal(combine_slots(c3, ids, "ep_normal", num_experts=E, ep_size=1), slot)  # one rank == EP=1 sum
    # the modes are observable: cancellation makes the slot order matter, bf16 partials make the rank grouping matter
    big = 2.0 ** 25
    xa = torch.tensor([[[big], [1.0], [-big], [1.0]]], device="cuda", dtype=torch.bfloat16)
    ia = torch.tensor([[8, 0, 12, 4]], device="cuda", dtype=torch.int32)
    assert combine_slots(xa, ia, "slot").item() == 1.0  # ((big + 1) - big) + 1: the first 1 is absorbed
    assert combine_slots(xa, ia, "expert_asc").item() == 0.0  # (1 + 1) + big - big: both ones absorbed
    xb = torch.tensor([[[1.0], [2.0 ** -9], [-1.0], [0.0]]], device="cuda", dtype=torch.bfloat16)
    ib = torch.tensor([[0, 1, 8, 12]], device="cuda", dtype=torch.int32)  # ranks 0, 0, 2, 3 of 4
    assert combine_slots(xb, ib, "slot").item() == 2.0 ** -9
    assert combine_slots(xb, ib, "ep_normal", num_experts=E, ep_size=4).item() == 0.0  # rank-0 partial 1 + 2^-9 rounds to 1.0 in bf16
    for m in COMBINE_MODES:
        assert combine_slots(c3, ids, m, num_experts=E, ep_size=ep).shape == (T, H)
