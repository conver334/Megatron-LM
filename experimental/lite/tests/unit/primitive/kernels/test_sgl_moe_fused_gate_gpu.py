# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Vendored SGLang moe_fused_gate: SGLang semantics (bias ranking, lowest-id ties, renorm, scale)
and how it relates to lite's torch routing."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

E, K, RSF = 128, 4, 2.0


def _torch_ref(logits, bias):
    """SGLang semantics in torch fp32: rank by sigmoid+bias with lowest-id ties, weight = sigmoid, renorm, x rsf."""
    sig = torch.sigmoid(logits)
    biased = sig + bias
    ids = torch.empty(logits.size(0), K, dtype=torch.long, device=logits.device)
    cur = biased.clone()
    ar = torch.arange(E, device=logits.device)
    for k in range(K):
        m = cur.max(dim=1, keepdim=True).values
        lane = torch.where(cur == m, ar, torch.full_like(ar, E + 1))
        win = lane.min(dim=1).values
        ids[:, k] = win
        cur[torch.arange(logits.size(0)), win] = -float("inf")
    sel = torch.gather(sig, 1, ids)
    return sel / sel.sum(1, keepdim=True) * RSF, ids


def test_matches_torch_semantics_and_breaks_ties_by_lowest_id():
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate

    torch.manual_seed(0)
    logits = torch.randn(512, E, device="cuda")
    bias = torch.randn(E, device="cuda") * 0.05
    w, ids = moe_fused_gate(logits, bias, K, routed_scaling_factor=RSF, apply_routed_scaling_factor_on_output=True)
    assert ids.dtype == torch.int32 and w.dtype == torch.float32
    rw, rids = _torch_ref(logits, bias)
    assert torch.equal(ids.long(), rids)
    torch.testing.assert_close(w, rw, rtol=1e-5, atol=1e-6)
    print(f"\nkernel vs torch weights bitwise mismatches: {(w != rw).sum().item()}/{w.numel()}")
    # exact ties need equal sigmoid+bias: zero bias, zero logits -> experts 0..3 (lowest id), weight rsf/K each
    tie = torch.zeros(8, E, device="cuda")
    tw, tids = moe_fused_gate(tie, torch.zeros(E, device="cuda"), K, routed_scaling_factor=RSF, apply_routed_scaling_factor_on_output=True)
    assert torch.equal(tids.long(), torch.arange(K, device="cuda").expand(8, K))
    assert torch.equal(tw, torch.full((8, K), RSF / K, device="cuda"))


def test_selection_set_equals_lite_routing_without_ties():
    from megatron.lite.primitive.kernels.sgl_moe_fused_gate import moe_fused_gate
    from megatron.lite.primitive.utils.moe import topk_routing_with_score_function

    torch.manual_seed(1)
    logits = torch.randn(1024, E, device="cuda")
    bias = torch.randn(E, device="cuda") * 0.05
    w, ids = moe_fused_gate(logits, bias, K, routed_scaling_factor=RSF, apply_routed_scaling_factor_on_output=True)
    probs, lite_ids = topk_routing_with_score_function(
        logits, K, score_function="sigmoid", expert_bias=bias, scaling_factor=RSF, dense_output=True
    )
    assert torch.equal(ids.long().sort(1).values, lite_ids.sort(1).values)
    order = ids.long().argsort(1)
    lite_sorted = torch.gather(probs, 1, lite_ids.argsort(1))
    kern_sorted = torch.gather(w, 1, order)
    diff = (kern_sorted - lite_sorted).abs()
    torch.testing.assert_close(kern_sorted, lite_sorted, rtol=1e-5, atol=1e-6)
    print(f"\nkernel vs lite weights: bitwise mismatches {(diff != 0).sum().item()}/{w.numel()}, max abs {diff.max().item():.2e}")
