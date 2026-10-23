# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TE linear wrappers whose forward shows SGLang's batch-invariant GEMM bytes (DeepGEMM bf16, see
``kernels.sgl_deepgemm.linear_bf16``) while the backward stays TE's. Applied by instance class swap so the
``linear.weight`` parameter and checkpoint names are untouched."""

from __future__ import annotations

import torch

from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
from megatron.lite.primitive.kernels.visible import visible_forward
from megatron.lite.primitive.parallel.linear import RowParallelLinear


class SglRowParallelLinear(RowParallelLinear):
    @classmethod
    def swap_in(cls, module: RowParallelLinear) -> "SglRowParallelLinear":
        if module.tp_size != 1 or getattr(module.linear, "use_bias", False):
            raise NotImplementedError("align linear: tp=1, no bias")
        module.__class__ = cls
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native = self.linear(x)
        return visible_forward(linear_bf16(x, self.linear.weight), native)
