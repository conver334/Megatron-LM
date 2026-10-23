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
    layers = _module("sglang.srt.layers")
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
