# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's bf16 MoE path under DeepEP normal mode (SGLANG_DEEPEP_BF16_DISPATCH=1, deterministic inference, the
DeepGEMM runner), one rank at a time: ep_scatter's layout (per-expert runs aligned to 128, zero-filled), DeepGEMM
m-grouped contiguous GEMMs, swigluoai, ep_gather (fp32 routed weight on the bf16 expert output, slot order) -> the
rank's bf16 partial; the DeepEP combine adds the partials of the ranks holding a slot (intranode: ascending rank in
fp32, one rounding; internode: each 8-rank node's sum rounded to bf16 first). Experts sit in contiguous blocks of
E / ep_size per rank (no EPLB)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module
from megatron.lite.primitive.kernels.sgl_swiglu import swigluoai

EXPERT_ALIGN = 128  # ep_scatter BLOCK_E == DeepEP expert_alignment under DeepGEMM
RANKS_PER_NODE = 8  # DeepEP NUM_MAX_NVL_PEERS


# --- vendored from the fork's kernels/ops/moe/ep_moe_kernels.py (text equality tested) ---
@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block_int32 = tl.program_id(0)
    cur_block = cur_block_int32.to(tl.int64)
    start_cur_token_int32 = tl.program_id(1)
    grid_num = tl.num_programs(1)
    for cur_token_int32 in range(start_cur_token_int32, total_token_num, grid_num):
        cur_token = cur_token_int32.to(tl.int64)
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index_int32 in range(0, topk_num):
            topk_index = topk_index_int32.to(tl.int64)
            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )
            if expert_id >= 0:
                source_token_index_int32 = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                source_token_index = source_token_index_int32.to(tl.int64)
                acc_weight = tl.load(
                    recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index
                )
                tmp = tl.load(
                    input_tensor
                    + source_token_index * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight
        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
):
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    BLOCK_D = 128 if hidden_size % 1024 != 0 else 1024  # block size of quantization
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


# --- end vendored ---


def rank_layout(local_ids: torch.Tensor, num_local_experts: int):
    """ep_scatter's layout for one rank (layout only, row order inside a run is free): rows grouped by local expert,
    each run padded to EXPERT_ALIGN. Returns (src token per row [total] (-1 = padding), m_indices int32 [total],
    output_index int64 [T, K] (-1 for remote slots), total)."""
    T, K = local_ids.shape
    flat = local_ids.reshape(-1).to(torch.int64)
    valid = flat >= 0
    counts = torch.bincount(flat[valid], minlength=num_local_experts)
    padded = (counts + EXPERT_ALIGN - 1) // EXPERT_ALIGN * EXPERT_ALIGN
    total = int(padded.sum().item())
    starts = torch.cumsum(padded, 0) - padded
    order = torch.argsort(torch.where(valid, flat, flat.new_full((), num_local_experts)), stable=True)[: int(valid.sum().item())]
    e_sorted = flat[order]
    dest = starts[e_sorted] + torch.arange(order.numel(), device=flat.device) - (torch.cumsum(counts, 0) - counts)[e_sorted]
    src = torch.full((total,), -1, dtype=torch.int64, device=flat.device)
    src[dest] = order // K
    output_index = torch.full_like(flat, -1)
    output_index[order] = dest
    m_indices = torch.repeat_interleave(torch.arange(num_local_experts, device=flat.device, dtype=torch.int32), padded)
    return src, m_indices, output_index.view(T, K), total


def rank_experts(hidden: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, local_ids: torch.Tensor,
                 topk_weights: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """One rank of the path: hidden [T, H] bf16, w13 [L, 2I, H], w2 [L, H, I], local_ids int [T, K] (-1 = expert on
    another rank), weights fp32 [T, K] -> bf16 partial [T, H] (+0.0 rows for tokens without a local slot)."""
    dg = deep_gemm_module()
    T, H = hidden.shape
    L, N, _ = w13.shape
    local_ids = local_ids.to(torch.int64)
    src, m_indices, output_index, total = rank_layout(local_ids, L)
    out = torch.zeros(T, H, dtype=hidden.dtype, device=hidden.device)
    if total == 0:
        return out
    a = torch.zeros(total, H, dtype=hidden.dtype, device=hidden.device)  # deterministic mode zero-fills the padding
    rows = src >= 0
    a[rows] = hidden[src[rows]]
    gateup = torch.empty(total, N, dtype=hidden.dtype, device=hidden.device)
    dg.m_grouped_bf16_gemm_nt_contiguous(a, w13, gateup, m_indices)
    act = swigluoai(gateup, alpha, limit)
    down = torch.empty(total, H, dtype=hidden.dtype, device=hidden.device)
    dg.m_grouped_bf16_gemm_nt_contiguous(act, w2, down, m_indices)
    ep_gather(down, local_ids, topk_weights.float().contiguous(), output_index, out)
    return out


def combine_rank_partials(partials: torch.Tensor, has_slot: torch.Tensor) -> torch.Tensor:
    """DeepEP normal-mode combine of per-rank bf16 partials [R, T, H]; has_slot [R, T] marks the ranks that received
    the token. R <= 8: ascending-rank fp32 sum from +0.0, one bf16 rounding; more ranks: each 8-rank node's sum rounded
    to bf16, nodes added in fp32 from +0.0 and rounded once more."""
    R, T, H = partials.shape
    n_nodes = (R + RANKS_PER_NODE - 1) // RANKS_PER_NODE
    acc = torch.zeros(T, H, dtype=torch.float32, device=partials.device)
    for node in range(n_nodes):
        lo, hi = node * RANKS_PER_NODE, min((node + 1) * RANKS_PER_NODE, R)
        node_acc = torch.zeros_like(acc)
        for r in range(lo, hi):
            node_acc = torch.where(has_slot[r][:, None], node_acc + partials[r].float(), node_acc)
        if n_nodes == 1:
            return node_acc.to(partials.dtype)
        acc = torch.where(has_slot[lo:hi].any(dim=0)[:, None], acc + node_acc.to(partials.dtype).float(), acc)
    return acc.to(partials.dtype)


def fused_experts_deepgemm_ep(hidden: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor,
                              topk_ids: torch.Tensor, alpha: float, limit: float, ep_size: int = 1) -> torch.Tensor:
    """The whole routed MoE as SGLang computes it over ep_size DeepEP ranks: hidden [T, H] bf16, w13 [E, 2I, H],
    w2 [E, H, I], weights fp32 [T, K], ids [T, K] -> [T, H] bf16 (before the separate shared-expert add)."""
    E = w13.shape[0]
    if ep_size < 1 or E % ep_size:
        raise ValueError(f"ep_size={ep_size} must divide the expert count {E}")
    L = E // ep_size
    ids = topk_ids.to(torch.int64)
    weights = topk_weights.float().contiguous()
    partials, has_slot = [], []
    for r in range(ep_size):
        local = (ids // L) == r
        local_ids = torch.where(local, ids - r * L, ids.new_full((), -1))
        partials.append(rank_experts(hidden, w13[r * L:(r + 1) * L], w2[r * L:(r + 1) * L], local_ids, weights, alpha, limit))
        has_slot.append(local.any(dim=1))
    return combine_rank_partials(torch.stack(partials), torch.stack(has_slot))
