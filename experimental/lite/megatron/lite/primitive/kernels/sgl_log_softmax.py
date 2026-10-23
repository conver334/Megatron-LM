# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang batch-invariant ``log_softmax`` (sglang 627bf1b2 srt/batch_invariant_ops/batch_invariant_ops.py,
``_log_softmax_kernel`` vendored verbatim) plus a stats twin with the same two reduction loops that stores
(max, logsumexp) per row, so the label's logprob ``(x - max) - lse`` costs O(rows) memory and is bitwise the
full kernel's element. This is the inference logprob contract (fp32 logits, original logprob, no temperature)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024  # as sglang log_softmax()


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    # Load first block to infer dtype and initialize max_val with correct type
    col_idx_init = tl.arange(0, BLOCK_SIZE)
    mask_init = col_idx_init < n_cols
    vals_init = tl.load(
        row_start_ptr + col_idx_init, mask=mask_init, other=-float("inf")
    )
    max_val = tl.max(vals_init)

    # Continue with remaining blocks
    for col_offset in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    # Initialize sum_exp with correct dtype by using tl.sum on a zero vector
    sum_exp = tl.sum(tl.zeros([1], dtype=max_val.dtype))

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


@triton.jit
def _log_softmax_stats_kernel(
    input_ptr,
    max_ptr,
    lse_ptr,
    input_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    # Load first block to infer dtype and initialize max_val with correct type
    col_idx_init = tl.arange(0, BLOCK_SIZE)
    mask_init = col_idx_init < n_cols
    vals_init = tl.load(
        row_start_ptr + col_idx_init, mask=mask_init, other=-float("inf")
    )
    max_val = tl.max(vals_init)

    # Continue with remaining blocks
    for col_offset in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    # Initialize sum_exp with correct dtype by using tl.sum on a zero vector
    sum_exp = tl.sum(tl.zeros([1], dtype=max_val.dtype))

    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    tl.store(max_ptr + row_idx, max_val)
    tl.store(lse_ptr + row_idx, log_sum_exp)


def log_softmax(x: torch.Tensor) -> torch.Tensor:
    """Full SGLang kernel over the last dim (fp32 in, fp32 out)."""
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty_like(x2)
    _log_softmax_kernel[(x2.shape[0],)](x2, out, x2.stride(0), out.stride(0), x2.shape[1], BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(x.shape)


def log_softmax_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    mx = torch.empty(x2.shape[0], dtype=x2.dtype, device=x2.device)
    lse = torch.empty_like(mx)
    _log_softmax_stats_kernel[(x2.shape[0],)](x2, mx, lse, x2.stride(0), x2.shape[1], BLOCK_SIZE=BLOCK_SIZE)
    return mx.reshape(x.shape[:-1]), lse.reshape(x.shape[:-1])


def token_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logprob of ``labels`` under ``log_softmax(logits)`` with the kernel's exact arithmetic: (x - max) - lse."""
    mx, lse = log_softmax_stats(logits)
    picked = torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1)
    return (picked - mx) - lse
