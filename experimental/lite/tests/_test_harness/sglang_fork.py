# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load SGLang fork source files by path with the sglang infrastructure stubbed, so cross-stack tests run the
fork's own router/GEMM code inside the training image. SGLANG_FORK_PY = <fork>/python/sglang."""

from __future__ import annotations

import contextlib
import functools
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


class _AttrModule(types.ModuleType):
    def __init__(self, name, values, default):
        super().__init__(name)
        self.__path__ = []
        self._values, self._default = values, default

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._values.get(name, self._default(name))


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
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
    import types as _types
    _exec = _types.SimpleNamespace(deterministic=_types.SimpleNamespace(enable_deterministic_inference=True),
                                   moe=_types.SimpleNamespace(enable_fused_moe_sum_all_reduce=False, ep_num_redundant_experts=0))
    sys.modules["sglang.srt.runtime_context"] = _AttrModule("sglang.srt.runtime_context", {"get_exec": lambda: _exec}, lambda name: (lambda *a, **k: None))
    utils._values.update({"get_device_name": lambda: "NVIDIA B200"})
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

    class MoeRunnerConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _module("sglang.srt.layers.moe.moe_runner", MoeRunnerConfig=MoeRunnerConfig)
    tu = _module("sglang.srt.layers.moe.moe_runner.triton_utils", get_config=lambda: None)
    tu.__path__ = [os.path.join(fork_root(), "srt/layers/moe/moe_runner/triton_utils")]  # relative imports hit the fork files
    _module("sglang.srt.utils.custom_op", register_custom_op=lambda *a, **k: (lambda fn: fn))
    return load("srt/layers/moe/moe_runner/triton_utils/fused_moe.py", "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe")
