# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Straight-through glue for kernel swaps: forward bytes from the inference kernel, backward from the native graph."""

from __future__ import annotations

import torch


class _VisibleForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
        return visible.view_as(visible)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, grad


def visible_forward(visible: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    """Returns ``visible``'s bytes (exactly, including -0.0/inf/NaN, no temporaries) with d(out)/d(native) == 1;
    ``visible`` gets no gradient. ``native`` must carry the autograd graph (TE / torch path)."""
    return _VisibleForward.apply(visible.detach(), native)
