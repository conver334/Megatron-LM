# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Stand-ins for the SGLang-side helpers the vendored FA4 package imports."""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def is_arch_support_pdl() -> bool:
    return torch.cuda.get_device_capability()[0] >= 9


def get_forward_host(*_a, **_k):  # SM120 dispatch is not vendored
    raise NotImplementedError("sgl_fa4: SM120 path not vendored")


def try_cached_paged_decode(*_a, **_k):
    return None


def try_cached_varlen(*_a, **_k):
    return None
