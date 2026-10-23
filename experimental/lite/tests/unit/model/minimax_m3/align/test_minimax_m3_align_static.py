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
    assert align_names - lite_names == {"align_modules", "align_router_gate", "align_moe_shared", "align_moe_combine", "align_moe_ep_size", "roll_labels"}
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


def test_align_swaps_are_applied_in_registry_order(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    calls = []
    monkeypatch.setattr(M.MiniMaxM3Layer, "__init__", lambda self, *a, **k: torch.nn.Module.__init__(self))
    monkeypatch.setitem(M.ALIGN_SWAPS, "a", lambda layer, cfg, ps, **o: calls.append(("a", o)))
    monkeypatch.setitem(M.ALIGN_SWAPS, "b", lambda layer, cfg, ps, **o: calls.append(("b", o)))
    layer = M._layer_factory(("b", "a"), {"router_gate": "fp32"})(SimpleNamespace(), SimpleNamespace(), 0)
    # registry order, not request order: "router" must be swapped before "experts"
    assert calls == [("a", {"router_gate": "fp32"}), ("b", {"router_gate": "fp32"})]
    assert layer.align_modules == ("b", "a")
    assert list(M.ALIGN_SWAPS).index("router") < list(M.ALIGN_SWAPS).index("experts")


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
    assert tc.align_modules == ("rmsnorm",) and tc.align_opts == {"router_gate": "fp32", "moe_shared": "separate", "moe_combine": "slot", "moe_ep_size": 1}
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


def test_swapped_attribute_names_exist_in_lite_source():
    """The swaps resolve attributes by name with getattr defaults; pin the names against lite's source."""
    import inspect

    from megatron.lite.model.minimax_m3.lite import model as LM

    layer_src = inspect.getsource(LM.MiniMaxM3Layer.__init__)
    assert "self.mlp_norm" in layer_src and "self.moe" in layer_src
    assert "self.norm" in inspect.getsource(LM.MiniMaxM3Model.__init__)
    assert "self._final_norm(" in inspect.getsource(LM.MiniMaxM3Model.forward)
    assert "self._token_loss(" in inspect.getsource(LM.MiniMaxM3Model.forward)
    assert "self.router" in inspect.getsource(LM.M3MoELayer.__init__)


def test_o_proj_swap_touches_every_attention_layer(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    seen = []
    fake = ModuleType("megatron.lite.primitive.parallel.linear_sgl")

    class SglRowParallelLinear:
        @classmethod
        def swap_in(cls, module):
            seen.append(module)
            return module

    fake.SglRowParallelLinear = SglRowParallelLinear
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    layers = [SimpleNamespace(attn=SimpleNamespace(proj=object())) for _ in range(3)]
    model = SimpleNamespace()
    for owner in (*layers, model):
        M.ALIGN_SWAPS["o_proj"](owner, SimpleNamespace(), None)
    assert seen == [layer.attn.proj for layer in layers]


def test_qkv_swap_touches_every_attention_layer(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    seen = []
    fake = ModuleType("megatron.lite.primitive.parallel.linear_sgl")

    class SglColumnParallelLinear:
        @classmethod
        def swap_in(cls, module):
            seen.append(module)
            return module

    fake.SglColumnParallelLinear = SglColumnParallelLinear
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    layers = [SimpleNamespace(attn=SimpleNamespace(qkv=object())) for _ in range(2)]
    for owner in (*layers, SimpleNamespace()):
        M.ALIGN_SWAPS["qkv"](owner, SimpleNamespace(), None)
    assert seen == [layer.attn.qkv for layer in layers]


def test_dense_mlp_swap_touches_dense_layers_only(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    seen = []
    fake = ModuleType("megatron.lite.model.minimax_m3.align.dense_mlp")

    class SglM3DenseMLP:
        @classmethod
        def swap_in(cls, module):
            seen.append(module)
            return module

    fake.SglM3DenseMLP = SglM3DenseMLP
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    dense = SimpleNamespace(mlp=object(), moe=None)
    moe = SimpleNamespace(mlp=None, moe=object())
    for owner in (dense, moe, SimpleNamespace()):
        M.ALIGN_SWAPS["dense_mlp"](owner, SimpleNamespace(), None)
    assert seen == [dense.mlp]


def test_lm_head_swap_touches_the_model_head_only(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    seen = []
    fake = ModuleType("megatron.lite.model.minimax_m3.align.lm_head")

    class SglVocabParallelOutput:
        @classmethod
        def swap_in(cls, module):
            seen.append(module)
            return module

    fake.SglVocabParallelOutput = SglVocabParallelOutput
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    model = SimpleNamespace(head=object())
    for owner in (SimpleNamespace(attn=object()), SimpleNamespace(head=None), model):
        M.ALIGN_SWAPS["lm_head"](owner, SimpleNamespace(), None)
    assert seen == [model.head]
    assert "self._token_loss(" in __import__("inspect").getsource(__import__("megatron.lite.model.minimax_m3.lite.model", fromlist=["x"]).MiniMaxM3Model.forward)


def test_prev_links_are_plain_references_not_submodules():
    from megatron.lite.model.minimax_m3.align import model as M

    a, b = torch.nn.Module(), torch.nn.Module()
    object.__setattr__(b, "_prev", a)  # the pattern MiniMaxM3AlignModel uses
    assert b._prev is a and "_prev" not in b._modules and len(list(b.children())) == 0
    assert 'object.__setattr__(layer, "_prev", prev)' in __import__("inspect").getsource(M.MiniMaxM3AlignModel.__init__)


def test_qkv_switch_refuses_pipeline_parallel():
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(NotImplementedError, match="pipeline"):
        A.build_model(MiniMaxM3Config(), impl_cfg=A.ImplConfig(parallel=ParallelConfig(pp=2), optimizer=None, align_modules=("qkv",)))


def test_experts_swap_touches_moe_layers_only(monkeypatch):
    from megatron.lite.model.minimax_m3.align import model as M

    seen = []
    fake = ModuleType("megatron.lite.model.minimax_m3.align.moe")

    class SglM3MoELayer:
        @classmethod
        def swap_in(cls, module, *, shared, combine, ep_size):
            seen.append((module, shared, combine, ep_size))
            return module

    fake.SglM3MoELayer = SglM3MoELayer
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    moe_layer = SimpleNamespace(moe=object())
    for owner in (moe_layer, SimpleNamespace(moe=None), SimpleNamespace()):
        M.ALIGN_SWAPS["experts"](owner, SimpleNamespace(), None, moe_shared="fused", moe_combine="expert_asc", moe_ep_size=8)
    assert seen == [(moe_layer.moe, "fused", "expert_asc", 8)]


def test_moe_shared_option_reaches_align_opts(monkeypatch):
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
    A.build_model(MiniMaxM3Config(), impl_cfg=A.ImplConfig(optimizer=None, align_modules=("router", "experts"), align_moe_shared="fused",
                                                            align_moe_combine="expert_asc", align_moe_ep_size=8))
    assert seen["train_cfg"].align_opts == {"router_gate": "fp32", "moe_shared": "fused", "moe_combine": "expert_asc", "moe_ep_size": 8}


def test_experts_swap_in_guards():
    from megatron.lite.model.minimax_m3.align.moe import SglM3MoELayer
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    def fake_moe(router_cls, ep=1):
        m = torch.nn.Module()
        m.dispatcher = SimpleNamespace(ep_size=ep)
        m.experts = SimpleNamespace(etp_group=None, num_local_experts=128)
        r = torch.nn.Module()
        r.gate = torch.nn.Module()
        r.__class__ = router_cls
        m.router = r
        return m

    with pytest.raises(ValueError, match="shared"):
        SglM3MoELayer.swap_in(fake_moe(SglSigmoidTopKRouter), shared="half")
    with pytest.raises(NotImplementedError, match="ep=1"):
        SglM3MoELayer.swap_in(fake_moe(SglSigmoidTopKRouter, ep=2))
    with pytest.raises(NotImplementedError, match="router switch"):
        SglM3MoELayer.swap_in(fake_moe(SigmoidTopKRouter))
    with pytest.raises(ValueError, match="combine"):
        SglM3MoELayer.swap_in(fake_moe(SglSigmoidTopKRouter), combine="pairwise")
    with pytest.raises(ValueError, match="EP=1"):
        SglM3MoELayer.swap_in(fake_moe(SglSigmoidTopKRouter), shared="fused", combine="ep_normal", ep_size=8)
    with pytest.raises(ValueError, match="divide"):
        SglM3MoELayer.swap_in(fake_moe(SglSigmoidTopKRouter), combine="ep_normal", ep_size=7)
    m = SglM3MoELayer.swap_in(fake_moe(SglSigmoidTopKRouter), shared="separate")
    assert isinstance(m, SglM3MoELayer) and (m.align_moe_shared, m.align_moe_combine, m.align_moe_ep_size) == ("separate", "slot", 1)
    m = SglM3MoELayer.swap_in(fake_moe(SglSigmoidTopKRouter), combine="ep_normal", ep_size=8)
    assert (m.align_moe_combine, m.align_moe_ep_size) == ("ep_normal", 8)
