# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static and CPU contracts of the MiniMax-M3 ``align`` implementation (lite + SGLang-kernel switches)."""

from __future__ import annotations

import sys
from dataclasses import fields
from types import ModuleType, SimpleNamespace

import pytest
import torch


def test_minimax_m3_registry_resolves_align_next_to_lite():
    from megatron.lite.model.registry import get_train_runtime_module, resolve_runtime_model_name

    assert resolve_runtime_model_name("minimax_m3", "lite") == "minimax_m3"
    assert resolve_runtime_model_name("minimax_m3", "align") == "minimax_m3_align"
    assert get_train_runtime_module("minimax_m3_align").__name__ == "megatron.lite.model.minimax_m3.align.protocol"


def test_align_protocol_exposes_the_runtime_surface():
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L

    for name in ("ImplConfig", "build_model_config", "build_model", "load_hf_weights", "export_hf_weights",
                 "save_hf_weights", "vocab_size", "EXPERT_CLASSIFIER", "PLACEMENT_FN", "unpack_forward_output"):
        assert hasattr(A, name), name
    # checkpoint mapping and forward step are shared with lite, not re-implemented
    assert A.load_hf_weights is L.load_hf_weights and A.export_hf_weights is L.export_hf_weights
    assert A.unpack_forward_output is L.unpack_forward_output


def test_align_impl_config_is_lite_plus_align_modules_with_training_recipe_defaults():
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L

    lite_names = {f.name for f in fields(L.ImplConfig)}
    align_names = {f.name for f in fields(A.ImplConfig)}
    assert align_names - lite_names == {"align_modules", "align_router_gate", "roll_labels"}
    cfg = A.ImplConfig()
    assert cfg.align_modules == ()
    for k, v in L.MINIMAX_TRAINING_IMPL.items():
        assert getattr(cfg, k) == v, k
    lite_cfg = A.to_lite_impl_cfg(cfg)
    assert isinstance(lite_cfg, L.ImplConfig) and lite_cfg.rope_fusion is True and lite_cfg.use_thd is False
    # verl_mlite sets use_thd=True; packed batches are native for M3, so align accepts it
    assert A.to_lite_impl_cfg(A.ImplConfig(use_thd=True)).use_thd is False


def test_align_layer_rejects_unknown_align_modules(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    monkeypatch.setattr(M.MiniMaxM3Layer, "__init__", lambda self, *a, **k: torch.nn.Module.__init__(self))
    M.MiniMaxM3AlignLayer(SimpleNamespace(), SimpleNamespace(), 0, align_modules=())
    with pytest.raises(ValueError, match="unknown align_modules"):
        M.MiniMaxM3AlignLayer(SimpleNamespace(), SimpleNamespace(), 0, align_modules=("no_such_module",))


def test_align_swaps_are_applied_in_order(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    calls = []
    monkeypatch.setattr(M.MiniMaxM3Layer, "__init__", lambda self, *a, **k: torch.nn.Module.__init__(self))
    monkeypatch.setitem(M.ALIGN_SWAPS, "a", lambda layer, cfg, ps, **o: calls.append(("a", o)))
    monkeypatch.setitem(M.ALIGN_SWAPS, "b", lambda layer, cfg, ps, **o: calls.append(("b", o)))
    layer = M._layer_factory(("b", "a"), {"router_gate": "fp32"})(SimpleNamespace(), SimpleNamespace(), 0)
    assert calls == [("b", {"router_gate": "fp32"}), ("a", {"router_gate": "fp32"})]
    assert layer.align_modules == ("b", "a")


def test_align_build_model_forwards_lite_cfg_and_align_modules(monkeypatch):
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
    from megatron.lite.model.minimax_m3.lite import protocol as L

    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, _cfg, train_cfg, _ps, **kwargs):
            super().__init__()
            seen["train_cfg"] = train_cfg
            self.layers = torch.nn.ModuleList()

        def to(self, *_a, **_k):
            return self

        def cuda(self, *_a, **_k):
            return self

    fake = ModuleType("megatron.lite.model.minimax_m3.align.model")
    fake.MiniMaxM3AlignModel = FakeModel
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    ps = SimpleNamespace(tp_size=1, ep_size=1, etp_size=1, pp_size=1, cp_size=1)
    monkeypatch.setattr(L, "init_parallel", lambda _cfg: ps)
    monkeypatch.setattr(L.magi_msa, "validate_device", lambda: None)
    monkeypatch.setattr(L.magi_msa, "validate_kernel_shapes", lambda **_k: None)
    monkeypatch.setattr(L.magi_msa, "build_msa_config", lambda *_a, **_k: object())

    bundle = A.build_model(MiniMaxM3Config(), impl_cfg=A.ImplConfig(optimizer=None, align_modules=("rmsnorm",)))
    assert bundle.forward_step is A._forward_step
    assert bundle.chunks[0].align_roll_labels is True
    tc = seen["train_cfg"]
    assert tc.align_modules == ("rmsnorm",) and tc.align_opts == {"router_gate": "bf16"}
    assert tc.rope_fusion is True and tc.msa_index_input_norm == "fused" and tc.moe_permute_fusion is False
    assert bundle.chunks[0].magi_settings.dense_kernel_backend == "fa4_varlen"


def test_align_rolls_labels_per_document_like_the_thd_packer():
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.runtime.contracts.data import PackedBatch

    batch = PackedBatch(
        input_ids=torch.tensor([10, 11, 12, 20, 21]),
        labels=torch.tensor([10, 11, 12, 20, 21]),
        loss_mask=torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0]),
        seq_lens=torch.tensor([3, 2]),
    )
    rolled = A.roll_packed_labels(batch)
    assert rolled.labels.tolist() == [11, 12, 0, 21, 0]
    assert rolled.loss_mask.tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]
    assert torch.equal(rolled.input_ids, batch.input_ids) and torch.equal(batch.labels, torch.tensor([10, 11, 12, 20, 21]))


def test_align_forward_step_rolls_only_when_enabled(monkeypatch):
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L
    from megatron.lite.runtime.contracts.data import PackedBatch

    seen = {}
    monkeypatch.setattr(L, "_forward_step", lambda model, batch: seen.setdefault("labels", batch.labels.tolist()))
    batch = PackedBatch(input_ids=torch.tensor([1, 2, 3]), labels=torch.tensor([1, 2, 3]), seq_lens=torch.tensor([3]))
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    model.align_roll_labels = True
    A._forward_step(model, batch)
    assert seen["labels"] == [2, 3, 0]
    seen.clear()
    model.align_roll_labels = False
    A._forward_step(model, batch)
    assert seen["labels"] == [1, 2, 3]


def test_rmsnorm_swap_replaces_standalone_norms_only(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    class FakeNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((8,), 0.5))

    swapped = []
    fake = ModuleType("megatron.lite.primitive.kernels.sgl_norm")

    class SglGemmaRMSNorm(torch.nn.Module):
        @classmethod
        def from_te(cls, module, eps):
            swapped.append((module, eps))
            return cls()

    fake.SglGemmaRMSNorm = SglGemmaRMSNorm
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    moe_layer = torch.nn.Module()
    moe_layer.mlp_norm = FakeNorm()
    dense_layer = torch.nn.Module()
    dense_layer.mlp_norm = None
    model = torch.nn.Module()
    model.norm = FakeNorm()
    cfg = SimpleNamespace(rms_norm_eps=1e-6)
    for owner in (moe_layer, dense_layer, model):
        M.ALIGN_SWAPS["rmsnorm"](owner, cfg, None)
    assert [eps for _, eps in swapped] == [1e-6, 1e-6]
    assert isinstance(moe_layer.mlp_norm, SglGemmaRMSNorm) and dense_layer.mlp_norm is None
    assert isinstance(model.norm, SglGemmaRMSNorm)


def test_router_swap_only_touches_moe_layers(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    seen = []
    fake = ModuleType("megatron.lite.primitive.modules.router_sgl")

    class SglSigmoidTopKRouter:
        @classmethod
        def swap_in(cls, router, *, gate):
            seen.append((router, gate))
            return router

    fake.SglSigmoidTopKRouter = SglSigmoidTopKRouter
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    moe_layer = SimpleNamespace(moe=SimpleNamespace(router=object()))
    dense_layer = SimpleNamespace(moe=None)
    model = SimpleNamespace()
    for owner in (moe_layer, dense_layer, model):
        M.ALIGN_SWAPS["router"](owner, SimpleNamespace(), None, router_gate="fp32")
    assert seen == [(moe_layer.moe.router, "fp32")]
