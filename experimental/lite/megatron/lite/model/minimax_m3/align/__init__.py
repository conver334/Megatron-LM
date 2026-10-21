# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 ``align`` implementation: the ``lite`` model with per-module switches that make the training forward
call the SGLang inference kernels (train-inference bitwise consistency). All switches off == ``lite``."""
