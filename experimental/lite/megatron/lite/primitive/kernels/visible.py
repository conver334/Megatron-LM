# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Straight-through glue for kernel swaps: forward bytes from the inference kernel, backward from the native graph."""

from __future__ import annotations

import torch


def visible_forward(visible: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    """``visible`` is what the forward shows (inference kernel output, no grad needed); ``native`` carries the
    autograd graph (TE / torch path). ``native - native.detach()`` is an exact zero, so the result is bitwise
    ``visible`` (except -0.0 -> +0.0) while d(result)/d(native) == 1."""
    return visible.detach() + (native - native.detach())
