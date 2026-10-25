# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load SGLang fork source files by path with the sglang infrastructure stubbed, so cross-stack tests run the
fork's own router/GEMM code inside the training image. SGLANG_FORK_PY = <fork>/python/sglang."""

from __future__ import annotations

import contextlib
import functools
import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest
import torch


def fork_root() -> str:
    root = os.environ.get("SGLANG_FORK_PY")
    if not root or not os.path.isdir(root):
        pytest.skip("SGLANG_FORK_PY (fork python/sglang dir) not set")
    return root


class _Env:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _set_path(m, path):
    """Make ``m`` a package rooted at ``path`` for the import system (``importlib.util.find_spec`` of submodules
    needs the parent's ``__spec__``)."""
    m.__path__ = list(path)
    m.__spec__ = importlib.machinery.ModuleSpec(m.__name__, None, is_package=True)
    m.__spec__.submodule_search_locations = m.__path__
    return m


class _AttrModule(types.ModuleType):
    def __init__(self, name, values, default):
        super().__init__(name)
        _set_path(self, [])
        self._values, self._default = values, default

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._values.get(name, self._default(name))


def _module(name, **attrs):
    m = types.ModuleType(name)
    _set_path(m, [])
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _bool_env(name, default="false"):
    return str(default).lower() in ("1", "true")


@functools.lru_cache(maxsize=None)
def install_stubs() -> bool:
    """Blackwell inference box: deterministic mode on, DeepGEMM on, PDL supported. Returns is_arch_support_pdl()."""
    pdl = torch.cuda.get_device_capability()[0] >= 9
    _module("sglang")
    _module("sglang.kernel_api_logging", debug_kernel_api=lambda fn: fn)
    _module("sglang.kernels")
    _module("sglang.kernels.jit")
    _module("sglang.kernels.jit.utils", cache_once=functools.lru_cache(maxsize=None), is_arch_support_pdl=lambda: pdl,
            load_jit=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA JIT path not under test")))
    _module("sglang.kernels.ops")
    _module("sglang.kernels.ops.moe")
    _module("sglang.kernels.ops.moe.moe_route_radix", covered=lambda *a, **k: False)
    _module("sglang.srt")
    envs = _AttrModule("sglang.srt.environ.envs",
                       {"SGLANG_OPT_BF16_FP32_GEMM_ALGO": _Env("deep_gemm"), "SGLANG_OPT_BF16_FP32_GEMM_CUTEDSL_MIN_M": _Env(4096)},
                       lambda name: _Env(False))
    _module("sglang.srt.environ", envs=envs)
    utils = _AttrModule("sglang.srt.utils", {"get_bool_env_var": _bool_env, "is_hip": lambda: False, "is_musa": lambda: False,
                                             "is_npu": lambda: False}, lambda name: (lambda *a, **k: False))
    sys.modules["sglang.srt.utils"] = utils
    sys.modules["sglang.srt.utils.common"] = _AttrModule(
        "sglang.srt.utils.common", {"get_bool_env_var": _bool_env, "get_dispatch_device_backend": lambda: "CUDA"},
        lambda name: (lambda *a, **k: False))
    _module("sglang.srt.server_args", ServerArgs=object)
    _module("sglang.srt.utils.custom_op", register_custom_op=lambda *a, **k: (lambda fn: fn))
    _module("sglang.utils", is_in_ci=lambda: False)
    _module("sglang.kernel_api_logging", debug_kernel_api=lambda fn: fn)
    import types as _types
    _exec = _types.SimpleNamespace(deterministic=_types.SimpleNamespace(enable_deterministic_inference=True),
                                   moe=_types.SimpleNamespace(enable_fused_moe_sum_all_reduce=False, ep_num_redundant_experts=0))
    sys.modules["sglang.srt.runtime_context"] = _AttrModule("sglang.srt.runtime_context", {"get_exec": lambda: _exec}, lambda name: (lambda *a, **k: None))
    utils._values.update({"get_device_name": lambda: "NVIDIA B200", "ceil_div": lambda x, y: (x + y - 1) // y})
    # fused_moe_triton_kernels.py: quant helpers are never reached for bf16; environment probes as on a CUDA box
    for q in ("sglang.kernels.ops.quantization", "sglang.kernels.ops.quantization.fp8_kernel", "sglang.kernels.ops.quantization.int8_kernel"):
        sys.modules[q] = _AttrModule(q, {}, lambda name: (lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"quant path {name} not under test"))))
    utils._values.update({"cpu_has_amx_support": lambda: False, "is_cpu": lambda: False, "is_cuda": lambda: True,
                          "is_sm90_supported": lambda: False, "is_xpu": lambda: False})
    sys.modules["sglang.srt.utils.nsys_profiler"] = _AttrModule(
        "sglang.srt.utils.nsys_profiler", {"nvtx_decorator": lambda *a, **k: (a[0] if a and callable(a[0]) and not k else (lambda fn: fn))},
        lambda name: (lambda *a, **k: None))
    sys.modules["sglang.srt.batch_invariant_ops"] = _AttrModule("sglang.srt.batch_invariant_ops", {"is_batch_invariant_mode_enabled": lambda: True}, lambda name: None)
    sys.modules["sglang.srt.layers.moe"] = _AttrModule("sglang.srt.layers.moe", {}, lambda name: None)
    sys.modules["sglang.srt.layers.moe.utils"] = _AttrModule("sglang.srt.layers.moe.utils", {"get_moe_padding_size": lambda aiter: 0}, lambda name: None)
    layers = _module("sglang.srt.layers")
    layers.moe = sys.modules["sglang.srt.layers.moe"]
    _module("sglang.srt.layers.deep_gemm_wrapper")
    _module("sglang.srt.layers.deep_gemm_wrapper.compile_utils",
            deep_gemm_execution_hook=lambda *a, **k: contextlib.nullcontext(),
            DeepGemmKernelType=_AttrModule("DeepGemmKernelType", {}, lambda name: name))
    sys.modules["sglang.srt.layers.deep_gemm_wrapper.configurer"] = _AttrModule(
        "sglang.srt.layers.deep_gemm_wrapper.configurer",
        {"ENABLE_JIT_DEEPGEMM": True, "DEEPGEMM_BLACKWELL": True, "DEEPGEMM_SCALE_UE8M0": True, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES": False},
        lambda name: False)
    entry = load("srt/layers/deep_gemm_wrapper/entrypoint.py", "sglang.srt.layers.deep_gemm_wrapper.entrypoint")
    sys.modules["sglang.srt.layers.deep_gemm_wrapper"] = entry
    layers.deep_gemm_wrapper = entry
    return pdl


def load(rel_path: str, modname: str):
    path = os.path.join(fork_root(), rel_path)
    if not os.path.exists(path):
        pytest.skip(f"fork file missing: {rel_path}")
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


class MoeRunnerConfig:
    """The fork's MoeRunnerConfig dataclass defaults (moe_runner/base.py), as a plain object."""

    def __init__(self, **kw):
        self.__dict__.update(dict(num_experts=None, num_local_experts=None, hidden_size=None, intermediate_size_per_partition=None,
                                  layer_id=None, top_k=None, num_fused_shared_experts=None, params_dtype=None, routing_method_type=None,
                                  activation="silu", is_gated=True, apply_router_weight_on_input=False, inplace=True, no_combine=False,
                                  routed_scaling_factor=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None, swiglu_limit=None,
                                  interleaved=True, gate_up_interleaved=True, layer=None, use_tp_all_gather_activation=False))
        self.__dict__.update(kw)


@functools.lru_cache(maxsize=None)
def load_fork_fused_moe():
    """The fork's ``fused_moe.py`` (``fused_experts_impl`` orchestration, M3's bf16 EP=1 path) with the pieces the
    training image cannot run stubbed: ``sgl_kernel.moe_sum_reduce`` -> the fork's own Triton ``moe_sum_reduce_triton``
    (an independent implementation of the same sum), ``sgl_kernel`` ``moe_align_block_size`` -> mlite's torch layout
    (layout-only, see the order-invariance test), symmetric memory / tp group / custom-op registration -> no-ops."""
    install_stubs()
    kernels = load("kernels/ops/moe/fused_moe_triton_kernels.py", "sglang.kernels.ops.moe.fused_moe_triton_kernels")
    _module("sglang.kernels.ops.activation")
    load("kernels/ops/activation/swigluoai_triton.py", "sglang.kernels.ops.activation.swigluoai_triton")
    _module("sglang.kernels.ops.activation.activation", gelu_and_mul=None, silu_and_mul=None)
    from megatron.lite.primitive.kernels.sgl_fused_moe import moe_align_block_size as _align

    def _sgl_align(topk_ids, num_experts_plus1, block_size, sorted_ids, expert_ids, num_tokens_post_pad, *_a, **_k):
        s, e, n = _align(topk_ids, block_size, num_experts_plus1 - 1)
        sorted_ids[: s.numel()].copy_(s)
        expert_ids[: e.numel()].copy_(e)
        num_tokens_post_pad.copy_(n)

    sys.modules["sglang.kernels.ops.moe"].moe_align_block_size = _sgl_align
    _module("sglang.kernels.ops.moe.moe_align_small_numel", SMALL_NUMEL_LIMIT=0, moe_align_small_numel=None)
    _module("sgl_kernel", moe_sum_reduce=kernels.moe_sum_reduce_triton)
    _module("sglang.srt.distributed", get_tp_group=lambda: None)
    _module("sglang.srt.distributed.device_communicators")
    _module("sglang.srt.distributed.device_communicators.pynccl_allocator", use_symmetric_memory=lambda *a, **k: contextlib.nullcontext())
    _module("sglang.srt.layers.dp_attention", is_allocation_symmetric=lambda: False)

    _module("sglang.srt.layers.moe.moe_runner", MoeRunnerConfig=MoeRunnerConfig)
    tu = _module("sglang.srt.layers.moe.moe_runner.triton_utils", get_config=lambda: None)
    _set_path(tu, [os.path.join(fork_root(), "srt/layers/moe/moe_runner/triton_utils")])  # relative imports hit the fork files
    return load("srt/layers/moe/moe_runner/triton_utils/fused_moe.py", "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe")


@functools.lru_cache(maxsize=None)
def load_fork_ep_kernels():
    """The fork's Triton EP kernels (ep_scatter / ep_gather / masked silu) by path: the DeepEP + DeepGEMM runner's
    per-rank glue, SGLang's own code."""
    install_stubs()
    return load("kernels/ops/moe/ep_moe_kernels.py", "sglang.kernels.ops.moe.ep_moe_kernels")


def _use_real_jit_utils():
    """Swap the stub ``sglang.kernels.jit.utils`` for the fork's real package (nvcc + ninja + tvm-ffi build of the
    CUDA JIT kernels); needs the fork tree's kernels/jit (csrc, include, utils) and msgspec."""
    import importlib

    root = fork_root()
    if not os.path.isdir(os.path.join(root, "kernels/jit/utils")):
        pytest.skip("fork kernels/jit tree not shipped")
    try:
        import msgspec  # noqa: F401
    except ImportError:
        pytest.skip("msgspec missing (fork JIT loader)")
    envs = sys.modules["sglang.srt.environ"].envs
    jit_cache = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "sgl_jit")
    envs._values.update({"SGLANG_JIT_CACHE_DIR": _Env(jit_cache), "SGLANG_JIT_CACHE_KEEP": _Env(None),
                         "SGLANG_JIT_CACHE_DEBUG": _Env(False), "SGLANG_JIT_KERNEL_RUN_FULL_TESTS": _Env(False)})
    for name in [n for n in sys.modules if n == "sglang.kernels.jit" or n.startswith("sglang.kernels.jit.")]:
        del sys.modules[name]
    jit_dir = os.path.join(root, "kernels/jit")
    spec = importlib.util.spec_from_file_location("sglang.kernels.jit", os.path.join(jit_dir, "__init__.py"),
                                                  submodule_search_locations=[jit_dir])  # a real package: find_spec().origin
    jit = importlib.util.module_from_spec(spec)
    sys.modules["sglang.kernels.jit"] = jit
    spec.loader.exec_module(jit)
    sys.modules["sglang.kernels"].jit = jit
    return importlib.import_module("sglang.kernels.jit.utils")


@functools.lru_cache(maxsize=None)
def load_fork_rope():
    """The fork's ``kernels/ops/attention/rope.py`` with its real CUDA JIT kernel (``elementwise/rope.cuh``)."""
    install_stubs()
    _use_real_jit_utils()
    return load("kernels/ops/attention/rope.py", "sglang.kernels.ops.attention.rope")


@functools.lru_cache(maxsize=None)
def load_fork_fa4():
    """The fork's vendored FA4 (``kernels/ops/attention/flash_attn/cute``, CuTe DSL with the batch-invariant switch)
    as a real package beside pip ``flash_attn.cute``."""
    import importlib

    install_stubs()
    root = fork_root()
    if not os.path.isdir(os.path.join(root, "kernels/ops/attention/flash_attn/cute")):
        pytest.skip("fork vendored FA4 not shipped")
    _use_real_jit_utils()  # the package resolves its JIT cache dir through the real kernels/jit tree
    _set_path(_module("sglang.kernels.ops.attention"), [os.path.join(root, "kernels/ops/attention")])
    sys.modules["sglang.kernels.ops.attention.fa4_sm120"] = _AttrModule("sglang.kernels.ops.attention.fa4_sm120", {}, lambda name: None)
    sys.modules["sglang.kernels.ops.attention.fa4_sm120.dispatch"] = _AttrModule(
        "sglang.kernels.ops.attention.fa4_sm120.dispatch", {}, lambda name: (lambda *a, **k: None))
    bi = importlib.import_module("sglang.kernels.ops.attention.flash_attn.cute.batch_invariance")
    iface = importlib.import_module("sglang.kernels.ops.attention.flash_attn.cute.interface")
    return bi, iface


@functools.lru_cache(maxsize=None)
def load_fork_deep_gemm_runner():
    """The fork's ``moe_runner/deep_gemm.py`` (DeepGemmRunnerCore and the deepep_normal pre/post permutes) by path:
    the swiglu comes from the fork's fused_moe module, ep_scatter/ep_gather from its EP kernels, DeepGEMM from its
    wrapper; registries, distributed and offloader helpers are no-ops."""
    import types as _t

    install_stubs()
    load_fork_fused_moe()
    load_fork_ep_kernels()
    try:
        import einops  # noqa: F401
    except ImportError:
        _module("einops", rearrange=None)
    sys.modules["sglang.kernels.ops.attention.dsv4"] = _AttrModule("sglang.kernels.ops.attention.dsv4", {"silu_and_mul_masked_post_quant": None}, lambda name: None)
    sys.modules["sglang.kernels.ops.quantization"]._values["per_token_group_quant"] = None

    class MoeRunnerCore:
        def __init__(self, config):
            self.config = config

    _module("sglang.srt.layers.moe.moe_runner.base", MoeQuantInfo=type("MoeQuantInfo", (), {}), RunnerInput=type("RunnerInput", (), {}),
            RunnerOutput=type("RunnerOutput", (), {}), MoeRunnerCore=MoeRunnerCore, MoeRunnerConfig=MoeRunnerConfig,
            register_pre_permute=lambda *a, **k: (lambda f: f), register_post_permute=lambda *a, **k: (lambda f: f))
    sys.modules["sglang.srt.layers.moe.utils"]._values["MoeRunnerBackend"] = _t.SimpleNamespace(DEEP_GEMM="deep_gemm")
    sys.modules["sglang.srt.utils"]._values["dispose_tensor"] = lambda *a, **k: None
    _module("sglang.srt.utils.offloader", get_offloader=lambda: _t.SimpleNamespace(forbid_copy_engine_usage=False))

    class DeepEPNormalCombineInput:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _module("sglang.srt.layers.moe.token_dispatcher")
    _module("sglang.srt.layers.moe.token_dispatcher.deepep", DeepEPNormalCombineInput=DeepEPNormalCombineInput,
            DeepEPLLCombineInput=DeepEPNormalCombineInput)
    return load("srt/layers/moe/moe_runner/deep_gemm.py", "sglang.srt.layers.moe.moe_runner.deep_gemm")


@functools.lru_cache(maxsize=None)
def load_fork_msa():
    """The fork's MiniMax sparse-attention prefill glue by path: ``minimax_sparse/prefill/cute_indexer.py`` (CuTe
    indexer GEMM from MM-Sparse-Attention ``inference.msa_v1`` + the CUDA radix top-k built by the fork's JIT loader)
    and ``cute_op_wrapper.py`` (gathered / paged sparse attention over the shared ``msa_v1`` kernels). Returns
    (cute_indexer, cute_op_wrapper)."""
    import importlib

    install_stubs()
    root = fork_root()
    for rel in ("kernels/ops/attention/minimax_sparse/prefill/cute_indexer.py", "kernels/jit/csrc/minimax/minimax_decode_topk.cuh"):
        if not os.path.exists(os.path.join(root, rel)):
            pytest.skip(f"fork file missing: {rel}")
    try:
        importlib.import_module("inference.msa_v1")
        importlib.import_module("msa_v1.attention")
    except ImportError as exc:
        pytest.skip(f"MM-Sparse-Attention packages missing: {exc}")
    _use_real_jit_utils()
    _set_path(_module("sglang.kernels.ops.attention"), [os.path.join(root, "kernels/ops/attention")])
    utils = sys.modules["sglang.srt.utils"]
    utils._values.update({"is_cuda": lambda: True, "is_hip": lambda: False, "is_npu": lambda: False})
    ci = importlib.import_module("sglang.kernels.ops.attention.minimax_sparse.prefill.cute_indexer")
    cw = importlib.import_module("sglang.kernels.ops.attention.minimax_sparse.prefill.cute_op_wrapper")
    if not (ci.CUTE_M3_INDEXER_AVAILABLE and cw.CUTE_AVAILABLE and cw.CUTE_PAGED_AVAILABLE):
        pytest.skip("fork CuTe MSA paths unavailable in this MM-Sparse-Attention build")
    return ci, cw
