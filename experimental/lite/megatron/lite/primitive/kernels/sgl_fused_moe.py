# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's Triton fused-MoE path for bf16 experts (sglang 627bf1b2 kernels/ops/moe/fused_moe_triton_kernels.py:
``write_zeros_to_output`` and ``fused_moe_kernel`` vendored verbatim), with the pieces of ``fused_experts`` that
matter for the M3 bf16 contract under --enable-deterministic-inference rewritten in torch:
``moe_align_block_size`` (layout only), the two kernel launches with the deterministic tile config
(BLOCK 64/64/32, GROUP 8, no TMA, no quant, swap_ab off), the shared Triton swigluoai in between, the routed weight
multiplied in fp32 inside the down GEMM, and ``moe_sum_reduce`` (sgl_kernel CUDA: fp32 sequential sum over the
top-k slots, bf16 round-to-nearest). Forward only."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from megatron.lite.primitive.kernels.sgl_swiglu import swigluoai

DETERMINISTIC_CONFIG = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    add_mask_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_bias_e,
    stride_bias_n,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
    swap_ab: tl.constexpr,
    FUSE_ADD_TO_OUTPUT: tl.constexpr,
    MASK_OUTPUT: tl.constexpr,
    FUSE_SUM_ALL_REDUCE: tl.constexpr,
    LORA_PRESERVE_BASE: tl.constexpr,
    ROUTER_TOPK: tl.constexpr,
    FUSE_SWIGLU: tl.constexpr = False,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)

    if filter_expert and off_experts == -1:
        if FUSE_SWIGLU:
            # C is the half-width post-activation buffer here. Rows owned by a
            # filtered expert are never read (the down-GEMM CTA for this block
            # early-exits before loading A), and an N-wide zero store would run
            # past the row end into a neighboring token's data.
            return
        if not FUSE_ADD_TO_OUTPUT and not (FUSE_SUM_ALL_REDUCE and LORA_PRESERVE_BASE):
            # Write zeros only when this kernel owns the full output; the experimental LoRA
            # add path (LORA_PRESERVE_BASE) keeps the base output from the prior MoE kernel.
            write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if a_desc is not None:
        assert use_fp8_w8a8 and group_n > 0 and group_k > 0
        start_offs_m = pid_m * BLOCK_SIZE_M
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    if b_desc is not None:
        start_offs_n = pid_n * BLOCK_SIZE_N
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if a_desc is not None:
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if a_desc is not None:
            a = a_desc.load([start_offs_m, k_start])
        elif even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if b_desc is not None:
            b = (
                b_desc.load([off_experts_i32, start_offs_n, k_start])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if swap_ab:
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    a_scale, b_scale = b_scale, a_scale
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                if use_fp8_w8a8:
                    if swap_ab:
                        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        if a_desc is None:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if b_desc is None:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))

    if use_int8_w8a16:
        accumulator *= b_scale
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k == 0 or group_n == 0:
            accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if FUSE_SWIGLU:
        # W13 rows were interleaved at load time, so gate/up of the same
        # intermediate channel sit in adjacent (even, odd) columns of this
        # tile; silu(gate) * up is applied in-register and only the
        # half-width activation is stored, eliminating intermediate_cache1
        # and the standalone activation launch.
        #
        # The asm below is the bit-parity contract with the `silu_and_mul` this
        # replaces, not an optimization: that kernel uses the fast-math
        # intrinsics `__fdividef(x, 1 + __expf(-x))`, while Triton's operators
        # lower to accurate expf and IEEE `div.rn`. The 1-2 ULP gap is enough to
        # flip the stored bf16 (8 mantissa bits) on many inputs. The final
        # multiply needs no asm -- plain fp32 multiply matches `mul.ftz.f32`
        # except on denormals. Silu stays fp32 until the store, as in the
        # reference; rounding it to bf16 first double-rounds and diverges.
        acc_pairs = tl.reshape(accumulator, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
        gate_b, up_b = tl.split(acc_pairs)
        gate_f = gate_b.to(tl.float32)
        # __expf(-x) == ex2.approx(x * -log2(e)); folding the negation into
        # the constant (0fBFB8AA3B == -log2(e)) flips the sign exactly.
        exp_neg = tl.inline_asm_elementwise(
            "{ mul.ftz.f32 $0, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 $0, $0; }",
            "=f,f",
            [gate_f],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        silu_f = tl.inline_asm_elementwise(
            "div.approx.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            [gate_f, 1.0 + exp_neg],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        out_act = (silu_f * up_b.to(tl.float32)).to(compute_type)
        offs_half = pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
        if c_sorted:
            c_ptrs = (
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_half[None, :]
            )
        else:
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_half[None, :]
            )
        c_mask = token_mask[:, None] & (offs_half[None, :] < N // 2)
        tl.store(c_ptrs, out_act, mask=c_mask)
    elif FUSE_ADD_TO_OUTPUT:
        # Accumulate into existing output with per-token mask.
        offs_token_out = offs_token // ROUTER_TOPK
        add_mask = tl.load(add_mask_ptr + offs_token_out, mask=token_mask, other=False)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & add_mask[:, None] & (offs_cn[None, :] < N)
        existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
        tl.store(c_ptrs, existing + accumulator, mask=c_mask)
    # ===== TO BE REFACTORED ====
    elif MASK_OUTPUT:
        # Store a fresh output while zeroing rows whose request has no active LoRA.
        offs_token_out = offs_token // ROUTER_TOPK
        output_mask = tl.load(
            add_mask_ptr + offs_token_out, mask=token_mask, other=False
        )
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        accumulator = tl.where(output_mask[:, None], accumulator, 0.0)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    # ===== END TO BE REFACTORED ====
    elif FUSE_SUM_ALL_REDUCE:
        offs_token_out = offs_token // ROUTER_TOPK
        c_ptrs = (
            c_ptr + stride_cm * offs_token_out[:, None] + stride_cn * offs_cn[None, :]
        )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
    else:
        if c_sorted:
            c_ptrs = (
                c_ptr
                + stride_cm * offs_token_id[:, None]
                + stride_cn * offs_cn[None, :]
            )
        else:
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """sgl_kernel moe_align_block_size semantics (layout only): the flattened (token, slot) indices sorted by expert,
    each expert's run padded to a multiple of ``block_size`` with the sentinel ``numel`` (ignored by the kernel),
    the expert id of every block, and the padded length."""
    flat = topk_ids.reshape(-1).to(torch.int64)
    numel = flat.numel()
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    starts = torch.cumsum(padded, 0) - padded
    total = int(padded.sum().item())
    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]
    rank_in_expert = torch.arange(numel, device=flat.device) - (torch.cumsum(counts, 0) - counts)[sorted_experts]
    sorted_ids = torch.full((total,), numel, dtype=torch.int32, device=flat.device)
    sorted_ids[starts[sorted_experts] + rank_in_expert] = order.to(torch.int32)
    expert_ids = torch.repeat_interleave(torch.arange(num_experts, device=flat.device), padded // block_size).to(torch.int32)
    num_tokens_post_pad = torch.tensor([total], dtype=torch.int32, device=flat.device)
    return sorted_ids, expert_ids, num_tokens_post_pad


def invoke_fused_moe_kernel_bf16(A, B, C, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
                                 mul_routed_weight: bool, top_k: int, config: dict = DETERMINISTIC_CONFIG, router_topk: int = 1) -> None:
    """The non-TMA, no-quant launch of ``invoke_fused_moe_kernel`` as SGLang issues it for bf16 under deterministic
    inference (filter_expert False: all experts local)."""
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16 and C.dtype == torch.bfloat16
    assert topk_weights.stride(1) == 1 and sorted_token_ids.stride(0) == 1
    K = B.shape[2]
    grid = (triton.cdiv(sorted_token_ids.shape[0], config["BLOCK_SIZE_M"]) * triton.cdiv(B.shape[1], config["BLOCK_SIZE_N"]),)
    fused_moe_kernel[grid](
        A, None, B, None, None, C, None, None, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, None,
        B.shape[1], K, sorted_token_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1), 0, 0, C.stride(-2), C.stride(-1),
        0, 0, 0, 0, 0,
        0, 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=tl.bfloat16,
        use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, per_channel_quant=False,
        even_Ks=(K % config["BLOCK_SIZE_K"] == 0), c_sorted=False, filter_expert=False, swap_ab=False,
        FUSE_ADD_TO_OUTPUT=False, MASK_OUTPUT=False, FUSE_SUM_ALL_REDUCE=False, LORA_PRESERVE_BASE=False,
        ROUTER_TOPK=router_topk, FUSE_SWIGLU=False,
        **config,
    )


COMBINE_MODES = ("slot", "expert_asc", "ep_normal")


def moe_sum_reduce(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """sgl_kernel / fork-Triton moe_sum_reduce: ``acc = 0; acc += float(x[:, k])`` in slot order, ``acc *= scale``,
    bf16 RN (the +0.0 start turns an all -0.0 row into +0.0, as the kernels do)."""
    acc = torch.zeros(x.shape[0], x.shape[2], dtype=torch.float32, device=x.device)
    for k in range(x.shape[1]):
        acc = acc + x[:, k].float()
    if scale != 1.0:
        acc = acc * scale
    return acc.to(x.dtype)


def combine_slots(cache3: torch.Tensor, topk_ids: torch.Tensor, mode: str = "slot", *, num_experts: int | None = None,
                  ep_size: int = 1, scale: float = 1.0) -> torch.Tensor:
    """[T, K, H] slot outputs (routed weights applied in the down GEMM) -> [T, H] bf16 in a declared accumulation order.
    slot: SGLang EP=1 Triton runner (K=1 written by the GEMM, K=2 one bf16 add, else moe_sum_reduce in slot order).
    expert_asc: slots summed in ascending expert id (the EP accumulation-order contract candidate).
    ep_normal: DeepEP normal-mode combine shape over ``ep_size`` ranks (experts in contiguous blocks): per rank a bf16
    partial of its slots in slot order with remote slots as zeros, ranks holding a slot added in ascending rank order in
    fp32 and rounded (intranode, <= 8 ranks); with more ranks each 8-rank node's sum is rounded to bf16 first and the
    nodes are added in fp32 (internode). The per-rank partial is the Triton-runner shape; the fork's own DeepEP path
    (DeepGEMM runner + ep_gather) weights the bf16 expert output after rounding, so only the cross-rank part has a
    fork counterpart."""
    K = cache3.shape[1]
    if mode == "slot":
        if K == 1 and scale == 1.0:
            return cache3[:, 0].clone()
        if K == 2 and scale == 1.0:
            return cache3[:, 0] + cache3[:, 1]
        return moe_sum_reduce(cache3, scale)
    ids = topk_ids.to(torch.int64)
    if mode == "expert_asc":
        order = torch.argsort(ids, dim=1, stable=True)
        return moe_sum_reduce(torch.gather(cache3, 1, order[:, :, None].expand_as(cache3)), scale)
    if mode == "ep_normal":
        from megatron.lite.primitive.kernels.sgl_deepgemm_moe import combine_rank_partials

        if num_experts is None or ep_size < 1 or num_experts % ep_size:
            raise ValueError(f"ep_normal needs num_experts divisible by ep_size, got {num_experts} / {ep_size}")
        ranks = ids // (num_experts // ep_size)
        zero = torch.zeros((), dtype=cache3.dtype, device=cache3.device)
        on = torch.stack([ranks == r for r in range(ep_size)])  # [R, T, K]
        partials = torch.stack([moe_sum_reduce(torch.where(on[r][:, :, None], cache3, zero), scale) for r in range(ep_size)])
        return combine_rank_partials(partials, on.any(dim=2))
    raise ValueError(f"combine mode {mode!r}; known: {COMBINE_MODES}")


def fused_experts_bf16(hidden: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor, alpha: float, limit: float, routed_scaling_factor: float = 1.0,
                       combine: str = "slot", ep_size: int = 1) -> torch.Tensor:
    """SGLang ``fused_experts`` for M3: hidden [T, H] bf16, w13 [E, 2I, H], w2 [E, H, I], weights fp32 [T, K],
    ids int32 [T, K] (SGLang slot order) -> [T, H] bf16; ``combine`` picks the slot accumulation order (combine_slots)."""
    T, H = hidden.shape
    E, N, _ = w13.shape
    K = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.float().contiguous()
    sorted_ids, expert_ids, n_post = moe_align_block_size(topk_ids, DETERMINISTIC_CONFIG["BLOCK_SIZE_M"], E)
    cache1 = torch.empty(T * K, N, dtype=hidden.dtype, device=hidden.device)
    invoke_fused_moe_kernel_bf16(hidden, w13, cache1, topk_weights, topk_ids, sorted_ids, expert_ids, n_post, False, K)
    cache2 = swigluoai(cache1, alpha, limit)
    cache3 = torch.empty(T, K, w2.shape[1], dtype=hidden.dtype, device=hidden.device)
    invoke_fused_moe_kernel_bf16(cache2, w2, cache3, topk_weights, topk_ids, sorted_ids, expert_ids, n_post, True, 1, router_topk=K)
    return combine_slots(cache3, topk_ids, combine, num_experts=E, ep_size=ep_size, scale=routed_scaling_factor)
