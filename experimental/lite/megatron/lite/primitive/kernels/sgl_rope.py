# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's RoPE as the fork applies it (``RotaryEmbedding.forward_cuda`` -> JIT ``rope.cuh``, neox style): a cos/sin
table computed on the CPU in fp32 exactly like ``_compute_cos_sin_cache`` (``inv_freq``, ``arange``, ``einsum``,
``cos``, ``sin``), the first ``rot`` dims of every head rotated in fp32 and stored back as bf16, the rest untouched.
TE's fused RoPE (the training backward) evaluates cos/sin inside the kernel, so its bytes differ."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


@functools.lru_cache(maxsize=None)
def _cpu_table(theta: float, rot: int, max_pos: int) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float) / rot))
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rot] fp32: cos | sin


class CosSinCache:
    """SGLang's ``cos_sin_cache`` for one (theta, rot), grown on demand in 4096-position steps; one GPU copy."""

    def __init__(self, theta: float, rot: int):
        self.theta, self.rot = float(theta), int(rot)
        self.table: torch.Tensor | None = None

    def rows(self, positions: torch.Tensor) -> torch.Tensor:
        need = int(positions.max().item()) + 1 if positions.numel() else 1
        if self.table is None or self.table.shape[0] < need or self.table.device != positions.device:
            size = (need + 4095) // 4096 * 4096
            self.table = _cpu_table(self.theta, self.rot, size).to(positions.device)
        return self.table


@triton.jit
def _rope_neox_kernel(x_ptr, out_ptr, cache_ptr, pos_ptr, stride_t, stride_h, n_heads, ROT: tl.constexpr, HALF: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    offs = tl.arange(0, HALF)
    base = x_ptr + t.to(tl.int64) * stride_t + h.to(tl.int64) * stride_h
    obase = out_ptr + t.to(tl.int64) * stride_t + h.to(tl.int64) * stride_h
    x = tl.load(base + offs).to(tl.float32)
    y = tl.load(base + HALF + offs).to(tl.float32)
    cos = tl.load(cache_ptr + pos * ROT + offs)
    sin = tl.load(cache_ptr + pos * ROT + HALF + offs)
    out_x = x * cos - y * sin
    out_y = x * sin + y * cos
    tl.store(obase + offs, out_x.to(out_ptr.dtype.element_ty))
    tl.store(obase + HALF + offs, out_y.to(out_ptr.dtype.element_ty))


def sgl_rope(x: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor, rot: int) -> torch.Tensor:
    """``x`` ``[T, H, D]`` bf16, ``positions`` ``[T]`` int, ``cache`` ``[>= max pos, rot]`` fp32 -> rotated copy."""
    T, H, D = x.shape
    out = x.clone()
    if T == 0:
        return out
    assert cache.dtype == torch.float32 and cache.shape[1] == rot and D >= rot and rot % 2 == 0
    _rope_neox_kernel[(T, H)](x, out, cache, positions, x.stride(0), x.stride(1), H, ROT=rot, HALF=rot // 2)
    return out
