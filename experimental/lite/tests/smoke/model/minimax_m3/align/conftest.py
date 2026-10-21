# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reuse the lite smoke fixtures (tiny HF-spelled config, torchrun process group, random HF source).
``tests`` is not a package, so the sibling conftest is loaded by path."""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "lite" / "conftest.py"
_spec = importlib.util.spec_from_file_location("minimax_m3_lite_smoke_conftest", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
