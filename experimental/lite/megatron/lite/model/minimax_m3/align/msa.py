# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MSA layers shown with SGLang's bytes: the indexer projection as the fork's fused [index_q | index_k] DeepGEMM
(``index_proj``) and the sparse core (indexer scores, top-k, block-sparse attention) in the inference ABI
(``msa_core``); Magi's ``calc_msa_v1`` keeps the backward and the KL loss."""

from __future__ import annotations

import torch

from megatron.lite.primitive.kernels import magi_msa
from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
from megatron.lite.primitive.kernels.sgl_msa import sgl_msa_prefill
from megatron.lite.primitive.kernels.visible import visible_forward
from megatron.lite.primitive.modules.attention.msa import MagiCoreAttention, MSAIndexer


class SglMSAIndexer(MSAIndexer):
    """Index projections through one fused DeepGEMM (SGLang ``index_qk_proj``, N = 4*128 + 128), split into q/k;
    norm and RoPE follow unchanged (TE RMSNorm == gemma_rmsnorm, RoPE via the ``_rope`` seam)."""

    @classmethod
    def swap_in(cls, module: MSAIndexer) -> "SglMSAIndexer":
        if module.q_proj.tp_size != 1:
            raise NotImplementedError("align index_proj: tp=1 only")
        module.__class__ = cls
        return module

    def index_qk(self, x_normed, freqs, positions=None):
        from megatron.lite.primitive.recompute import is_recomputing

        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            x_normed = x_normed.detach()
            S, B = x_normed.shape[:2]
            nq = self.num_heads * self.head_dim
            iq = self.q_proj(x_normed)
            ik = self.k_proj(x_normed)
            if not is_recomputing():
                with torch.no_grad():
                    w = torch.cat([self.q_proj.linear.weight, self.k_proj.weight], dim=0)
                    fused = linear_bf16(x_normed.reshape(-1, x_normed.shape[-1]), w)
                iq = visible_forward(fused[:, :nq].reshape(iq.shape), iq)
                ik = visible_forward(fused[:, nq:].reshape(ik.shape), ik)
            iq = self._rope(self.q_norm(iq.view(S, B, -1, self.head_dim)), freqs, positions)
            ik = self._rope(self.k_norm(ik.view(S, B, 1, self.head_dim)), freqs, positions)
        return iq, ik


class SglMagiCoreAttention(MagiCoreAttention):
    """Visible bytes = SGLang's prefill chain on the packed documents (inference ABI), native = Magi ``calc_msa_v1``."""

    def forward(self, q, k, v, index_q, index_k, *, magi_ctx):
        from megatron.lite.primitive.recompute import is_recomputing

        native = magi_msa.calc_msa_v1(q, k, v, index_q, index_k, magi_ctx)
        if is_recomputing():
            return native
        if magi_ctx.cp_size != 1:
            raise NotImplementedError("align msa_core: cp=1 (the visible path recomputes the packed documents locally)")
        with torch.no_grad():
            packed = [magi_msa.undispatch_tokens(t.detach(), magi_ctx) for t in (q, k, v, index_q, index_k)]
            out = sgl_msa_prefill(*packed, cu_seqlens=list(magi_ctx.cu_seqlens_host))
            visible = magi_msa.dispatch_tokens(out, magi_ctx)
        return visible_forward(visible.view_as(native), native)


def swap_in_index_proj(attn) -> None:
    indexer = getattr(attn, "indexer", None)
    if indexer is not None:
        SglMSAIndexer.swap_in(indexer)


def swap_in_msa_core(attn) -> None:
    core = getattr(attn, "core_attn", None)
    if isinstance(core, MagiCoreAttention):
        core.__class__ = SglMagiCoreAttention
