# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's DeepEP-normal + DeepGEMM bf16 MoE path (sgl_deepgemm_moe): ep_gather kernel == fork, one rank's partial
== the fork's own kernels composed as its DeepGEMM runner composes them (ep_scatter -> grouped_gemm_nt_bf16_contig ->
swigluoai -> grouped_gemm_nt_bf16_contig -> ep_gather), layout / padding / token-order invariance, DeepEP combine
reference (intranode and internode), fp64 sanity and EP datums."""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

T, H, I, E, K = 300, 512, 256, 16, 4
ALPHA, LIMIT = 1.702, 7.0


def _inputs(seed=0, t=T, h=H, i_=I, e=E):
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate

    torch.manual_seed(seed)
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    w13 = (torch.randn(e, 2 * i_, h, device="cuda", dtype=torch.bfloat16) * 0.02)
    w2 = (torch.randn(e, h, i_, device="cuda", dtype=torch.bfloat16) * 0.02)
    weights, ids = moe_fused_gate(torch.randn(t, e, device="cuda"), torch.randn(e, device="cuda") * 0.05, K,
                                  routed_scaling_factor=2.0, apply_routed_scaling_factor_on_output=True)
    return x, w13, w2, weights, ids


def _fork():
    from sglang_fork import install_stubs, load, load_fork_ep_kernels

    install_stubs()
    ep = load_fork_ep_kernels()
    entry = sys.modules["sglang.srt.layers.deep_gemm_wrapper"]
    name = "sglang.kernels.ops.activation.swigluoai_triton"
    swig = sys.modules.get(name) or load("kernels/ops/activation/swigluoai_triton.py", name)
    return ep, entry, swig


def _fork_rank_partial(x, w13_r, w2_r, local_ids, weights, perm):
    """What the fork computes on one DeepEP rank (pre_permute_deepep_normal_to_deep_gemm -> _run_bf16_contiguous_gemm
    with the swigluoai activation -> post_permute ep_gather), for the tokens ``perm`` in that order."""
    ep, entry, swig = _fork()
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module

    deep_gemm_module()  # deterministic flag, as SGLang's batch-invariant mode sets it
    L, N, h = w13_r.shape
    recv_x, recv_ids, recv_w = x[perm].contiguous(), local_ids[perm].contiguous(), weights[perm].contiguous()
    counts = torch.bincount(recv_ids[recv_ids >= 0], minlength=L)
    aligned = ((counts + 127) // 128 * 128).to(torch.int32)  # DeepEP expert_alignment=128 under DeepGEMM
    all_tokens = int(aligned.sum().item())
    input_tensor = torch.zeros(all_tokens, h, dtype=torch.bfloat16, device="cuda")  # deterministic: zero-filled
    input_scale = torch.zeros(all_tokens, h // 128, dtype=torch.float32, device="cuda")
    m_indices = torch.zeros(all_tokens, dtype=torch.int32, device="cuda")
    output_index = torch.empty_like(recv_ids)
    start = torch.empty_like(aligned)
    ep.ep_scatter(recv_x, None, recv_ids, aligned, aligned, start, input_tensor, input_scale, m_indices, output_index,
                  scale_ue8m0=False, quant_block_size=128)
    gateup = torch.empty(all_tokens, N, dtype=torch.bfloat16, device="cuda")
    entry.grouped_gemm_nt_bf16_contig(input_tensor, w13_r, gateup, m_indices)
    down_input = swig.swigluoai(gateup.view(-1, N), ALPHA, LIMIT).contiguous()
    down = torch.empty(all_tokens, h, dtype=torch.bfloat16, device="cuda")
    entry.grouped_gemm_nt_bf16_contig(down_input, w2_r, down, m_indices)
    gather = torch.empty(perm.numel(), h, dtype=torch.bfloat16, device="cuda")
    ep.ep_gather(down, recv_ids, recv_w, output_index, gather)
    return gather


def _fork_rank_partial_runner(x, w13_r, w2_r, local_ids, weights, perm):
    """The same rank through the fork's own runner code: pre_permute_deepep_normal_to_deep_gemm ->
    DeepGemmRunnerCore.run (bf16 contiguous path, oai-swiglu) -> post_permute_deep_gemm_to_deepep_normal."""
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module
    from sglang_fork import MoeRunnerConfig, load_fork_deep_gemm_runner

    deep_gemm_module()
    dg = load_fork_deep_gemm_runner()
    L = w13_r.shape[0]
    recv_x, recv_ids, recv_w = x[perm].contiguous(), local_ids[perm].contiguous(), weights[perm].contiguous()
    counts = torch.bincount(recv_ids[recv_ids >= 0], minlength=L)
    aligned = ((counts + 127) // 128 * 128).tolist()  # DeepEP's num_recv_tokens_per_expert_list (expert_alignment=128)
    cfg = MoeRunnerConfig(activation="silu", is_gated=True, gemm1_alpha=ALPHA, gemm1_clamp_limit=LIMIT, gate_up_interleaved=False,
                          interleaved=False, inplace=False, top_k=recv_ids.shape[1], num_experts=L, num_local_experts=L)
    quant = dg.DeepGemmMoeQuantInfo(w13_weight=w13_r, w2_weight=w2_r, use_fp8=False)
    state = {}
    rin = dg.pre_permute_deepep_normal_to_deep_gemm((recv_x, None, recv_ids, recv_w, aligned, None), quant, cfg, state)
    rout = dg.DeepGemmRunnerCore(cfg).run(rin, quant, state)
    return dg.post_permute_deep_gemm_to_deepep_normal(rout, quant, cfg, state).hidden_states


def test_ep_gather_kernel_equals_fork_and_launch():
    from megatron.lite.primitive.kernels import sgl_deepgemm_moe as V

    ep, _, _ = _fork()
    ours, theirs = inspect.getsource(V._fwd_kernel_ep_gather.fn), inspect.getsource(ep._fwd_kernel_ep_gather.fn)
    assert ast.dump(ast.parse(textwrap.dedent(ours))) == ast.dump(ast.parse(textwrap.dedent(theirs)))
    torch.manual_seed(2)
    rows = torch.randn(T * K, H, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(-1, 4, (T, K), device="cuda", dtype=torch.int64)
    index = torch.randint(0, T * K, (T, K), device="cuda", dtype=torch.int64)
    w = torch.rand(T, K, device="cuda")
    a, b = torch.empty(T, H, device="cuda", dtype=torch.bfloat16), torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
    V.ep_gather(rows, ids, w, index, a)
    ep.ep_gather(rows, ids, w, index, b)
    assert torch.equal(a, b)


@pytest.mark.parametrize("impl", ["composed", "runner"])
@pytest.mark.parametrize("shape", ["unit", "model_width"])
def test_rank_partial_equals_fork_runner_composition(shape, impl):
    """One DeepEP rank: mlite rank_experts == the fork's ep_scatter + grouped_gemm_nt_bf16_contig + swigluoai +
    ep_gather run by path on the rank's tokens in a shuffled arrival order ("composed": the kernels called as the
    runner calls them; "runner": the fork's pre_permute / DeepGemmRunnerCore / post_permute functions themselves). Bitwise."""
    from megatron.lite.primitive.kernels.sgl_deepgemm_moe import rank_experts

    if shape == "unit":
        x, w13, w2, weights, ids = _inputs(1)
        ep_size, r = 4, 1
    else:
        x, w13, w2, weights, ids = _inputs(3, t=1024, h=6144, i_=3072, e=32)
        ep_size, r = 4, 2
    e = w13.shape[0]
    L = e // ep_size
    ids64 = ids.to(torch.int64)
    local = (ids64 // L) == r
    local_ids = torch.where(local, ids64 - r * L, ids64.new_full((), -1))
    ours = rank_experts(x, w13[r * L:(r + 1) * L], w2[r * L:(r + 1) * L], local_ids, weights, ALPHA, LIMIT)
    sel = local.any(dim=1).nonzero().flatten()
    perm = sel[torch.randperm(sel.numel(), generator=torch.Generator(device="cuda").manual_seed(0), device="cuda")]
    fork_fn = _fork_rank_partial if impl == "composed" else _fork_rank_partial_runner
    theirs = fork_fn(x, w13[r * L:(r + 1) * L], w2[r * L:(r + 1) * L], local_ids, weights, perm)
    assert torch.equal(theirs, ours[perm]), f"{(theirs != ours[perm]).float().mean().item():.4f} of the rank's bytes differ"
    assert (ours[~local.any(dim=1)] == 0).all() and ours.abs().max().item() > 0


def test_rank_partial_is_layout_padding_and_token_set_invariant(monkeypatch):
    from megatron.lite.primitive.kernels import sgl_deepgemm_moe as V

    x, w13, w2, weights, ids = _inputs(4)
    L = E // 2
    ids64 = ids.to(torch.int64)
    local = (ids64 // L) == 0
    local_ids = torch.where(local, ids64, ids64.new_full((), -1))
    full = V.rank_experts(x, w13[:L], w2[:L], local_ids, weights, ALPHA, LIMIT)
    sel = local.any(dim=1).nonzero().flatten()
    subset = V.rank_experts(x[sel], w13[:L], w2[:L], local_ids[sel], weights[sel], ALPHA, LIMIT)
    assert torch.equal(subset, full[sel])  # the rank only ever sees its own tokens
    monkeypatch.setattr(V, "EXPERT_ALIGN", 256)
    assert torch.equal(V.rank_experts(x, w13[:L], w2[:L], local_ids, weights, ALPHA, LIMIT), full)  # padding is not in the bytes


def test_combine_rank_partials_reference_intranode_and_internode():
    from megatron.lite.primitive.kernels.sgl_deepgemm_moe import combine_rank_partials

    torch.manual_seed(5)
    for R in (4, 16):
        partials = torch.randn(R, T, H, device="cuda", dtype=torch.bfloat16)
        has = torch.rand(R, T, device="cuda") < 0.3
        has[0] |= ~has.any(dim=0)  # every token on at least one rank
        ref = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
        for t in range(T):
            acc = torch.zeros(H, device="cuda")
            for node in range((R + 7) // 8):
                node_acc = torch.zeros(H, device="cuda")
                for r in range(node * 8, min(node * 8 + 8, R)):
                    if has[r, t]:
                        node_acc = node_acc + partials[r, t].float()
                if R <= 8:
                    acc = node_acc
                elif has[node * 8:node * 8 + 8, t].any():
                    acc = acc + node_acc.to(torch.bfloat16).float()
            ref[t] = acc.to(torch.bfloat16)
        assert torch.equal(combine_rank_partials(partials, has), ref), R
    # observable: node-level bf16 rounding (ranks 0, 1, 2 on node 0 hold 1, 2^-9, 2^-9; rank 8 on node 1 holds 2^-9)
    p = torch.zeros(16, 1, 1, device="cuda", dtype=torch.bfloat16)
    p[0], p[1], p[2], p[8] = 1.0, 2.0 ** -9, 2.0 ** -9, 2.0 ** -9
    has = torch.zeros(16, 1, dtype=torch.bool, device="cuda")
    has[[0, 1, 2, 8]] = True
    assert combine_rank_partials(p, has).item() == 1.0  # node 0: 1 + 2^-8 -> bf16 1.0; + 2^-9 -> 1.0
    assert combine_rank_partials(p[:9].clone().index_fill_(0, torch.tensor([8], device="cuda"), 0)[:8], has[:8]).item() == 1.0
    flat = (p[0].float() + p[1].float() + p[2].float() + p[8].float()).to(torch.bfloat16).item()
    assert flat == 1.0 + 2.0 ** -7  # one fp32 pass would round up instead


def test_fused_experts_deepgemm_ep_sanity_and_datums():
    from megatron.lite.primitive.kernels.sgl_deepgemm_moe import combine_rank_partials, fused_experts_deepgemm_ep, rank_experts
    from megatron.lite.primitive.kernels.sgl_fused_moe import fused_experts_bf16

    x, w13, w2, weights, ids = _inputs(6)
    ref = torch.zeros(T, H, device="cuda", dtype=torch.float64)
    for k in range(K):
        e = ids[:, k].long()
        gu = torch.einsum("th,tnh->tn", x.double(), w13.double()[e])
        gate, up = gu.chunk(2, dim=-1)
        act = gate.clamp(max=LIMIT) * torch.sigmoid(gate.clamp(max=LIMIT) * ALPHA) * (up.clamp(-LIMIT, LIMIT) + 1)
        ref += torch.einsum("ti,thi->th", act, w2.double()[e]) * weights[:, k].double()[:, None]
    outs = {}
    for ep in (1, 4, 16):
        outs[ep] = fused_experts_deepgemm_ep(x, w13, w2, weights, ids, ALPHA, LIMIT, ep_size=ep)
        assert torch.equal(outs[ep], fused_experts_deepgemm_ep(x, w13, w2, weights, ids, ALPHA, LIMIT, ep_size=ep))
        torch.testing.assert_close(outs[ep].double(), ref, rtol=3e-2, atol=3e-2)
    one = rank_experts(x, w13, w2, ids.to(torch.int64), weights, ALPHA, LIMIT)
    assert torch.equal(outs[1], combine_rank_partials(one[None], torch.ones(1, T, dtype=torch.bool, device="cuda")))
    assert torch.equal(outs[1], one)  # a single rank: +0.0 + partial, rounding is the identity
    triton = fused_experts_bf16(x, w13, w2, weights, ids, ALPHA, LIMIT)
    print(f"\nDeepEP+DeepGEMM path: EP=1 vs EP=4 {(outs[1] != outs[4]).float().mean().item():.4f} of bytes differ, "
          f"EP=1 vs EP=16 {(outs[1] != outs[16]).float().mean().item():.4f}; EP=1 vs Triton fused_moe path "
          f"{(outs[1] != triton).float().mean().item():.4f}, max |d| {(outs[1].float() - triton.float()).abs().max().item():.3e}", flush=True)
    with pytest.raises(ValueError):
        fused_experts_deepgemm_ep(x, w13, w2, weights, ids, ALPHA, LIMIT, ep_size=3)
    with pytest.raises(ValueError, match="multiple"):
        combine_rank_partials(torch.zeros(12, 2, 4, device="cuda", dtype=torch.bfloat16), torch.ones(12, 2, dtype=torch.bool, device="cuda"))
