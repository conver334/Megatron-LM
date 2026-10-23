# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LM head and token logprob shown as SGLang computes them: bf16 logits from the batch-invariant GEMM, fp32
``log_softmax`` from the batch-invariant Triton kernel, gathered at the label. Native paths (vanilla matmul,
vocab-parallel cross entropy) stay the backward."""

from __future__ import annotations

import torch

from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
from megatron.lite.primitive.kernels.sgl_log_softmax import token_logprob
from megatron.lite.primitive.kernels.visible import visible_forward
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.parallel.linear import VocabParallelOutput
from megatron.lite.primitive.recompute import is_recomputing


class SglVocabParallelOutput(VocabParallelOutput):
    @classmethod
    def swap_in(cls, module: VocabParallelOutput) -> "SglVocabParallelOutput":
        if module.col.linear.weight.shape[0] != module.vocab_size:
            raise NotImplementedError("align lm_head: tp=1 (unpadded vocab) only")
        module.__class__ = cls
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native = self.col.linear(x)  # first: the owning module's forward pre-hook finishes the DDP param gather
        if is_recomputing():
            return native
        return visible_forward(linear_bf16(x, self.col.linear.weight), native)


def sgl_token_loss(logits: torch.Tensor, labels_sb: torch.Tensor, tp_group) -> torch.Tensor:
    """-logprob[label] with SGLang bytes (logits.float() -> BI log_softmax), gradient of the native CE."""
    native = vocab_parallel_cross_entropy(logits, labels_sb, tp_group)
    with torch.no_grad():
        visible = -token_logprob(logits.detach(), labels_sb)  # bf16 logits upcast inside the kernel
    return visible_forward(visible, native)
