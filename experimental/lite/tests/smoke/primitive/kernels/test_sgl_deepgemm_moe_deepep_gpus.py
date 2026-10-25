# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real DeepEP (intranode, normal mode) vs the modeled DeepEP path: every rank dispatches its own tokens through
DeepEP, runs mlite's rank_experts on what it receives, combines through DeepEP, and the result must equal
fused_experts_deepgemm_ep(ep_size=world) computed locally from the rank's original tokens. Run with
torchrun --nproc_per_node N (N in 2/4/8 on one node); needs deep_ep importable (intranode-only build is enough)."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(2)]

ALPHA, LIMIT = 1.702, 7.0


def _dist():
    import torch.distributed as dist

    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("needs torchrun with >= 2 ranks")
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    return dist


@pytest.mark.parametrize("shape", ["unit", "model_width"])
def test_real_deepep_combine_equals_modeled_path(shape):
    dist = _dist()
    deep_ep = pytest.importorskip("deep_ep")
    from megatron.lite.primitive.kernels.sgl_deepgemm_moe import fused_experts_deepgemm_ep, rank_experts
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate

    world, rank = dist.get_world_size(), dist.get_rank()
    group = dist.group.WORLD
    if shape == "unit":
        H, I, E, K, T = 512, 256, 16, 4, 300 + 7 * rank
    else:
        H, I, E, K, T = 6144, 3072, 32, 4, 512 + 31 * rank
    L = E // world
    torch.manual_seed(100)  # identical experts on every rank
    w13 = (torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.02)
    w2 = (torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.02)
    torch.manual_seed(1000 + rank)  # each rank's own tokens
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    weights, ids = moe_fused_gate(torch.randn(T, E, device="cuda"), torch.randn(E, device="cuda") * 0.05, K,
                                  routed_scaling_factor=2.0, apply_routed_scaling_factor_on_output=True)
    ids64 = ids.to(torch.int64)
    ours = fused_experts_deepgemm_ep(x, w13, w2, weights, ids, ALPHA, LIMIT, ep_size=world)

    deep_ep.Buffer.set_num_sms(24)
    hidden_bytes = H * 2
    nvl = max(cfg.get_nvl_buffer_size_hint(hidden_bytes, world)
              for cfg in (deep_ep.Buffer.get_dispatch_config(world), deep_ep.Buffer.get_combine_config(world)))
    buffer = deep_ep.Buffer(group, nvl, 0)
    n_rank, n_rdma, n_expert, in_rank, _ = buffer.get_dispatch_layout(ids64, E)
    recv_x, recv_ids, recv_w, n_recv_per_expert, handle, _ = buffer.dispatch(
        x, topk_idx=ids64, topk_weights=weights, num_tokens_per_rank=n_rank, num_tokens_per_rdma_rank=n_rdma,
        is_token_in_rank=in_rank, num_tokens_per_expert=n_expert, expert_alignment=128)
    assert all(n % 128 == 0 for n in n_recv_per_expert)  # SGLang's pre-permute relies on the aligned counts
    partial = rank_experts(recv_x, w13[rank * L:(rank + 1) * L], w2[rank * L:(rank + 1) * L], recv_ids, recv_w, ALPHA, LIMIT)
    combined, _, _ = buffer.combine(partial, handle)
    torch.cuda.synchronize()
    same = torch.equal(combined, ours)
    frac = (combined != ours).float().mean().item()
    flags = [None] * world
    dist.all_gather_object(flags, (rank, same, frac, T, recv_x.shape[0]))
    if rank == 0:
        print(f"\n[{shape}] world={world} E={E}: " + "; ".join(f"rank {r}: {'equal' if s else f'{f:.4f} differ'} ({t} own / {rt} recv tokens)" for r, s, f, t, rt in flags), flush=True)
    dist.barrier()
    assert all(s for _, s, *_ in flags), flags
    assert ours.abs().max().item() > 0
