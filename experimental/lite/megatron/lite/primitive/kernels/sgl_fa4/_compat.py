# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Stand-ins for the SGLang-side helpers the vendored FA4 package imports."""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def is_arch_support_pdl() -> bool:
    return torch.cuda.get_device_capability()[0] >= 9


def get_forward_host(arch: int):
    """fa4_sm120.dispatch.get_forward_host: an architecture-owned host only on SM120 (not vendored), else None."""
    if arch // 10 == 12:
        raise NotImplementedError("sgl_fa4: SM120 forward host not vendored")
    return None


def try_cached_paged_decode(*, arch: int, **_k):
    return None if get_forward_host(arch) is None else None


def try_cached_varlen(*, arch: int, **_k):
    return None if get_forward_host(arch) is None else None
