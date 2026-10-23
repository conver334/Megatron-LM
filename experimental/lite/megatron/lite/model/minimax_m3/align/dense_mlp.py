# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense MLP (layers 0-2) shown as SGLang computes it: post_attention_layernorm = fused add+norm of the
(attn_out, residual) pair, gate_up GEMM and down GEMM through DeepGEMM, swigluoai through the shared Triton
kernel. The TE graph (LayerNormLinear -> quick_geglu -> Linear) stays the backward path."""

from __future__ import annotations

import torch

from megatron.lite.model.minimax_m3.lite.model import M3DenseMLP
from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
from megatron.lite.primitive.kernels.sgl_norm import visible_input_norm
from megatron.lite.primitive.kernels.sgl_swiglu import swigluoai
from megatron.lite.primitive.kernels.visible import visible_forward
from megatron.lite.primitive.recompute import is_recomputing


class SglM3DenseMLP(M3DenseMLP):
    _residual_pair = None  # (attn_out, residual) set by the align layer before the call

    @classmethod
    def swap_in(cls, module: M3DenseMLP) -> "SglM3DenseMLP":
        if module.gate_up.tp_size != 1 or module.config.swiglu_up_offset != 1.0:
            raise NotImplementedError("align dense_mlp: tp=1 and swigluoai (up + 1) only")
        module.__class__ = cls
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native = super().forward(x)
        if is_recomputing():
            return native
        with torch.no_grad():
            normed = visible_input_norm(x, self.gate_up.linear.layer_norm_weight, self.gate_up.linear.eps, self._residual_pair)
            act = swigluoai(linear_bf16(normed, self.gate_up.linear.weight), self.config.swiglu_alpha, self.config.swiglu_limit)
            out = linear_bf16(act, self.down.linear.weight)
        return visible_forward(out, native)
