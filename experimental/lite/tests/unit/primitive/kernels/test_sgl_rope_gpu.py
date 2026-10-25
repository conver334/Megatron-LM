# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang RoPE (sgl_rope): the cos/sin table is SGLang's CPU recipe, the rotate equals the fork's CUDA JIT kernel
run by path, positions index the table, the non-rotary dims pass through; TE fused RoPE (training) differs."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

T, H, D, ROT, THETA = 2048, 64, 128, 64, 5_000_000.0


def _inputs(seed=0, t=T):
    torch.manual_seed(seed)
    x = torch.randn(t, H, D, device="cuda", dtype=torch.bfloat16)
    pos = torch.randint(0, 65536, (t,), device="cuda", dtype=torch.int64)
    return x, pos


def test_table_is_sglang_cpu_recipe_and_cache_grows():
    from megatron.lite.primitive.kernels.sgl_rope import CosSinCache, _cpu_table

    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float) / ROT))  # RotaryEmbedding._compute_inv_freq
    t = torch.arange(8192, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    ref = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    assert torch.equal(_cpu_table(THETA, ROT, 8192), ref)
    gpu_freqs = torch.outer(torch.arange(8192, device="cuda", dtype=torch.float32), inv_freq.cuda())
    print(f"\ncos/sin table CPU (SGLang) vs GPU torch.cos/sin of the same freqs: {(ref.cuda() != torch.cat((gpu_freqs.cos(), gpu_freqs.sin()), -1)).float().mean().item():.4f} differ")
    cache = CosSinCache(THETA, ROT)
    pos = torch.tensor([5, 4099], device="cuda")
    rows = cache.rows(pos)
    assert rows.shape[0] == 8192 and torch.equal(rows[:8192], ref.cuda())
    assert cache.rows(torch.tensor([3], device="cuda")) is rows  # no regrowth for smaller positions


def test_rope_rotates_first_dims_and_leaves_the_rest():
    from megatron.lite.primitive.kernels.sgl_rope import CosSinCache, sgl_rope

    x, pos = _inputs(1)
    cache = CosSinCache(THETA, ROT)
    out = sgl_rope(x, pos, cache.rows(pos), ROT)
    assert torch.equal(out[..., ROT:], x[..., ROT:]) and not torch.equal(out[..., :ROT], x[..., :ROT])
    assert torch.equal(out, sgl_rope(x, pos, cache.rows(pos), ROT))
    zero = torch.zeros_like(pos)
    assert torch.equal(sgl_rope(x, zero, cache.rows(zero), ROT), x)  # position 0: cos 1, sin 0


def test_rope_equals_fork_jit_kernel_by_path():
    """The fork's ``apply_rope_inplace`` (CUDA JIT ``rope.cuh``, neox) on q and k == sgl_rope, bitwise."""
    from megatron.lite.primitive.kernels.sgl_rope import CosSinCache, sgl_rope
    from sglang_fork import load_fork_rope

    rope = load_fork_rope()
    x, pos = _inputs(2)
    k = torch.randn(T, 4, D, device="cuda", dtype=torch.bfloat16)
    cache = CosSinCache(THETA, ROT).rows(pos)
    q_f, k_f = x.clone(), k.clone()
    rope.apply_rope_inplace(q_f[..., :ROT], k_f[..., :ROT], cache, pos, is_neox=True)  # strided views as forward_cuda passes them
    ours_q = sgl_rope(x, pos, cache, ROT)
    ours_k = sgl_rope(k, pos, cache, ROT)
    assert torch.equal(ours_q, q_f), f"q: {(ours_q != q_f).float().mean().item():.5f} differ"
    assert torch.equal(ours_k, k_f), f"k: {(ours_k != k_f).float().mean().item():.5f} differ"


def test_te_fused_rope_differs_datum():
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

    from megatron.lite.primitive.kernels.sgl_rope import CosSinCache, sgl_rope

    x, _ = _inputs(3)
    pos = torch.arange(T, device="cuda")
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float32, device="cuda") / ROT))
    f = torch.outer(pos.float(), inv)
    freqs = torch.cat((f, f), dim=-1)[:, None, None, :]  # lite RotaryEmbedding.get_emb: [S, 1, 1, rot] non-interleaved
    te = apply_rotary_pos_emb(x[:, None], freqs, tensor_format="sbhd", interleaved=False, fused=True)[:, 0]
    ours = sgl_rope(x, pos, CosSinCache(THETA, ROT).rows(pos), ROT)
    assert torch.equal(te[..., ROT:], ours[..., ROT:])
    torch.testing.assert_close(ours.float(), te.float(), rtol=2 ** -5, atol=2 ** -5)
    print(f"\nTE fused RoPE vs SGLang RoPE: {(te[..., :ROT] != ours[..., :ROT]).float().mean().item():.4f} of rotary bytes differ, "
          f"max |d| {(te.float() - ours.float()).abs().max().item():.3e}")
