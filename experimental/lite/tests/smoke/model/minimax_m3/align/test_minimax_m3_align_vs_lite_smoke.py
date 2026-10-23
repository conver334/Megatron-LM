# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``align`` with no align_modules must be bitwise identical to ``lite`` under the same ImplConfig knobs
(forward logits/loss and every parameter gradient), on one GPU, single and packed documents.
A lite A/A run is the control: grads that are not run-to-run reproducible on this image (Magi builds without
deterministic MSAV1 backward) are only required to stay within the A/A noise."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"), pytest.mark.optional]

CHUNK = 512
BATCHES = {"single": [4096], "packed": [1536, 2560]}


def _magi_deps():
    pytest.importorskip("magi_attn_extensions.MSA")
    pytest.importorskip("msa_v1")


def _deterministic_supported() -> bool:
    from megatron.lite.primitive.kernels import magi_msa

    try:
        magi_msa.build_msa_config(magi_msa.MagiMsaSettings(deterministic=True))
    except ValueError:
        return False
    return True


def _dense_backend() -> str:
    """MINIMAX_TRAINING_IMPL uses fa4_varlen (pip flash_attn.cute); fall back to Magi's fa4 where it is absent (OCI venv)."""
    try:
        import flash_attn.cute  # noqa: F401
    except ImportError:
        return "fa4"
    return "fa4_varlen"


def _impl_cfg(proto, **kw):
    from megatron.lite.runtime.contracts import ParallelConfig

    kw.setdefault("magi_dense_kernel_backend", _dense_backend())
    return proto.ImplConfig(parallel=ParallelConfig(), optimizer=None, deterministic=_deterministic_supported(),
                            magi_chunk_size=CHUNK, **kw)


def _run(proto, cfg, src, seq_lens, impl_cfg, *, pre_roll=False, hooks=None):
    from megatron.lite.model.minimax_m3.align.protocol import roll_packed_labels
    from megatron.lite.runtime.contracts.data import PackedBatch

    torch.manual_seed(7)
    bundle = proto.build_model(cfg, impl_cfg=impl_cfg)
    chunk = bundle.chunks[0]
    proto.load_hf_weights(chunk, src, cfg, bundle.parallel_state)
    if hooks is not None:
        hooks(chunk)
    routing = {}
    for i, layer in enumerate(chunk.layers):
        if layer.moe is not None:
            layer.moe.router.register_forward_hook(
                lambda m, inp, out, i=i: routing.__setitem__(i, (out[0].detach().float().clone(), out[1].detach().clone()))
            )
    torch.manual_seed(1234)
    total = sum(seq_lens)
    ids = torch.randint(0, cfg.vocab_size, (total,), device="cuda")
    labels = torch.randint(0, cfg.vocab_size, (total,), device="cuda")
    batch = PackedBatch(input_ids=ids, labels=labels, loss_mask=torch.ones(total, device="cuda"), seq_lens=torch.tensor(seq_lens))
    if pre_roll:  # lite takes pre-rolled labels; align rolls them itself (roll_labels=True)
        batch = roll_packed_labels(batch)
    out = bundle.forward_step(chunk, batch)
    loss = out["loss"]
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in chunk.named_parameters() if p.grad is not None}
    logits = out.get("logits")
    result = dict(loss=loss.detach().clone(), grads=grads, logits=None if logits is None else logits.detach().clone(),
                  routing=routing, log_probs=out["log_probs"].detach().clone())
    del bundle, chunk
    torch.cuda.empty_cache()
    return result


@pytest.mark.parametrize("tag", list(BATCHES))
def test_align_without_switches_matches_lite_bitwise(magi_cfg, magi_source, dist, tag):
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L

    _magi_deps()
    seq_lens = BATCHES[tag]
    knobs = {**L.MINIMAX_TRAINING_IMPL, "magi_dense_kernel_backend": _dense_backend()}
    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **knobs), pre_roll=True)
    lite2 = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **knobs), pre_roll=True)  # A/A control
    align = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=()))
    assert torch.equal(lite["loss"], lite2["loss"]), "lite forward is not reproducible on this image"
    assert torch.equal(lite["loss"], align["loss"]), (lite["loss"].item(), align["loss"].item())
    if lite["logits"] is not None:
        assert torch.equal(lite["logits"], lite2["logits"])
        assert torch.equal(lite["logits"], align["logits"])
    assert lite["grads"].keys() == align["grads"].keys()

    def _rel(a, b):
        return ((a.float() - b.float()).abs().max() / a.float().abs().max().clamp_min(1e-30)).item()

    noise = {n: _rel(lite["grads"][n], lite2["grads"][n]) for n in lite["grads"]}
    diff = {n: _rel(lite["grads"][n], align["grads"][n]) for n in lite["grads"]}
    if _deterministic_supported():
        # only the dense fa4 backward is non-reproducible (atomics). Noise is judged per module group (layers.N / embed /
        # norm / head): a group is stable when all its grads are bitwise in lite A/A (a single param can be equal by
        # chance); stable groups must be bitwise in align, the rest stay inside 2x the A/A spread
        group = lambda n: ".".join(n.split(".")[:2]) if n.startswith("layers.") else n.split(".")[0]
        unstable_groups = {group(n) for n, v in noise.items() if v > 0.0}
        stable = [n for n in noise if group(n) not in unstable_groups]
        assert stable, "no reproducible grads under Magi deterministic mode"
        bad = [n for n in stable if diff[n] != 0.0]
        assert bad == [], bad[:10]
        bound = 2 * max(noise.values())
        mode = f"deterministic, {len(stable)} grads in stable groups bitwise, noisy groups {sorted(unstable_groups)}"
    else:
        # MSAV1 backward itself is non-reproducible: lite-vs-align relative spread must sit inside the lite A/A spread
        bound = 2 * max(noise.values())
        assert bound > 0, "lite A/A grads all bitwise but Magi reports no deterministic mode; revisit the criterion"
        mode = "Magi deterministic unsupported"
    over = {n: (diff[n], noise[n]) for n in diff if diff[n] > bound}
    assert over == {}, (bound, dict(list(over.items())[:10]))
    bitwise = sum(1 for n in diff if diff[n] == 0.0)
    print(f"\n{tag}: forward bitwise; grads {bitwise}/{len(diff)} bitwise, max rel diff {max(diff.values()):.2e} "
          f"vs A/A bound {bound:.2e} ({mode}, dense {_dense_backend()})", flush=True)


def test_control_unrolled_labels_change_the_loss(magi_cfg, magi_source, dist):
    """Planted error: align with roll_labels=False on unrolled labels must not match lite (criterion can FAIL)."""
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L

    _magi_deps()
    seq_lens = BATCHES["packed"]
    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **{**L.MINIMAX_TRAINING_IMPL, "magi_dense_kernel_backend": _dense_backend()}), pre_roll=True)
    align = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=(), roll_labels=False))
    assert not torch.equal(lite["loss"], align["loss"])


def test_rmsnorm_switch_runs_and_only_perturbs_by_norm_rounding(magi_cfg, magi_source, dist):
    """align_modules=("rmsnorm",): standalone norms run the SGLang kernel; loss finite, close to lite,
    and the swap is visible (planted change) on MoE layers and the final norm."""
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L
    from megatron.lite.primitive.kernels.sgl_norm import SglGemmaRMSNorm, gemma_rmsnorm_kernel

    _magi_deps()
    try:
        gemma_rmsnorm_kernel()
    except ImportError as exc:
        pytest.skip(str(exc))
    seq_lens = BATCHES["packed"]
    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **{**L.MINIMAX_TRAINING_IMPL, "magi_dense_kernel_backend": _dense_backend()}), pre_roll=True)
    torch.manual_seed(7)
    bundle = A.build_model(magi_cfg, impl_cfg=_impl_cfg(A, align_modules=("rmsnorm",)))
    chunk = bundle.chunks[0]
    n_sgl = sum(isinstance(m, SglGemmaRMSNorm) for m in chunk.modules())
    n_moe = sum(1 for layer in chunk.layers if layer.moe is not None)
    assert n_sgl == n_moe + 1, (n_sgl, n_moe)
    del bundle, chunk
    torch.cuda.empty_cache()
    align = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=("rmsnorm",)))
    assert torch.isfinite(align["loss"]).all()
    rel = ((lite["loss"] - align["loss"]).abs() / lite["loss"].abs()).item()
    assert rel < 1e-2, rel
    assert not torch.equal(lite["loss"], align["loss"]), "rmsnorm switch had no visible effect"
    print(f"\nrmsnorm switch: {n_sgl} norms swapped, loss rel diff vs lite {rel:.2e}", flush=True)


def test_router_switch_matches_lite_routing_and_is_visible(magi_cfg, magi_source, dist):
    """align_modules=("router",) alone: expert sets equal lite's on every MoE layer (fp32 gate, same inputs),
    scores are the kernel's bytes, and the bf16 gate (SGLang default) is a visible change."""
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L
    from megatron.lite.primitive.modules.router_sgl import SglSigmoidTopKRouter

    _magi_deps()
    seq_lens = BATCHES["packed"]
    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **{**L.MINIMAX_TRAINING_IMPL, "magi_dense_kernel_backend": _dense_backend()}), pre_roll=True)
    torch.manual_seed(7)
    bundle = A.build_model(magi_cfg, impl_cfg=_impl_cfg(A, align_modules=("router",)))
    chunk = bundle.chunks[0]
    n_sgl = sum(isinstance(m, SglSigmoidTopKRouter) for m in chunk.modules())
    n_moe = sum(1 for layer in chunk.layers if layer.moe is not None)
    assert n_sgl == n_moe > 0, (n_sgl, n_moe)
    del bundle, chunk
    torch.cuda.empty_cache()
    fp32 = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=("router",), align_router_gate="fp32"))
    bf16 = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=("router",), align_router_gate="bf16"))
    for res in (fp32, bf16):
        assert torch.isfinite(res["loss"]).all() and all(torch.isfinite(g).all() for g in res["grads"].values())
    assert lite["routing"].keys() == fp32["routing"].keys() == bf16["routing"].keys()
    for i in lite["routing"]:
        (ls, li), (fs, fi), (bs, bi) = lite["routing"][i], fp32["routing"][i], bf16["routing"][i]
        same = (li.sort(1).values == fi.sort(1).values).all(1).float().mean().item()
        ws = torch.gather(fs, 1, fi.argsort(1)) - torch.gather(ls, 1, li.argsort(1))
        print(f"\nlayer {i}: lite vs align(fp32 gate) same expert set {same:.4f} (first-layer inputs identical only "
              f"for the first MoE layer), weight max |d| {ws.abs().max().item():.2e}; bf16 vs fp32 gate ids differ on "
              f"{(bi.sort(1).values != fi.sort(1).values).any(1).float().mean().item():.4f} of rows", flush=True)
        if i == min(lite["routing"]):
            assert same > 0.999, same  # same inputs, same fp32 gate math up to GEMM rounding -> only near-ties may move
    rel = ((lite["loss"] - fp32["loss"]).abs() / lite["loss"].abs()).item()
    assert rel < 1e-2, rel
    assert not torch.equal(lite["loss"], bf16["loss"]), "bf16 gate switch had no visible effect"
    print(f"\nrouter switch: {n_sgl} routers swapped, loss rel diff vs lite fp32 gate {rel:.2e}, "
          f"bf16 gate {((lite['loss'] - bf16['loss']).abs() / lite['loss'].abs()).item():.2e}", flush=True)


def test_o_proj_switch_runs_close_to_lite_and_is_visible(magi_cfg, magi_source, dist):
    """align_modules=("o_proj",): every attention output projection shows DeepGEMM bytes; TE backward unchanged."""
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module
    from megatron.lite.primitive.parallel.linear_sgl import SglRowParallelLinear

    _magi_deps()
    try:
        deep_gemm_module()
    except ImportError as exc:
        pytest.skip(str(exc))
    seq_lens = BATCHES["packed"]
    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **{**L.MINIMAX_TRAINING_IMPL, "magi_dense_kernel_backend": _dense_backend()}), pre_roll=True)
    torch.manual_seed(7)
    bundle = A.build_model(magi_cfg, impl_cfg=_impl_cfg(A, align_modules=("o_proj",)))
    chunk = bundle.chunks[0]
    n_sgl = sum(isinstance(m, SglRowParallelLinear) for m in chunk.modules())
    assert n_sgl == len(chunk.layers), (n_sgl, len(chunk.layers))
    del bundle, chunk
    torch.cuda.empty_cache()
    align = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=("o_proj",)))
    assert torch.isfinite(align["loss"]).all() and all(torch.isfinite(g).all() for g in align["grads"].values())
    rel = ((lite["loss"] - align["loss"]).abs() / lite["loss"].abs()).item()
    assert rel < 1e-2, rel
    # DeepGEMM and TE (cuBLAS) may agree bitwise on a given shape, so the loss change is a report, not a gate;
    # the swap itself is pinned by the module count and by test_sgl_linear_gpu.py
    print(f"\no_proj switch: {n_sgl} projections swapped, loss rel diff vs lite {rel:.2e} "
          f"(bitwise equal to lite: {torch.equal(lite['loss'], align['loss'])})", flush=True)


def test_qkv_switch_runs_close_to_lite(magi_cfg, magi_source, dist):
    """align_modules=("rmsnorm", "qkv"): every attention input norm + QKV shows SGLang bytes (fused add+norm of
    the previous layer's residual pair, DeepGEMM); residual pairs are handed layer to layer and cleared."""
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module
    from megatron.lite.primitive.parallel.linear_sgl import SglColumnParallelLinear

    _magi_deps()
    try:
        deep_gemm_module()
    except ImportError as exc:
        pytest.skip(str(exc))
    seq_lens = BATCHES["packed"]
    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **{**L.MINIMAX_TRAINING_IMPL, "magi_dense_kernel_backend": _dense_backend()}), pre_roll=True)
    torch.manual_seed(7)
    bundle = A.build_model(magi_cfg, impl_cfg=_impl_cfg(A, align_modules=("rmsnorm", "qkv")))
    chunk = bundle.chunks[0]
    assert sum(isinstance(m, SglColumnParallelLinear) for m in chunk.modules()) == len(chunk.layers)
    assert all(layer._prev is prev for prev, layer in zip(chunk.layers, list(chunk.layers)[1:]))
    assert all("_prev" not in layer._modules for layer in chunk.layers)
    del bundle, chunk
    torch.cuda.empty_cache()
    # pair handoff: layer 0 sees no pair, every later layer sees its predecessor's (delta, residual) and its QKV
    # bytes are the fused add+norm of that pair through DeepGEMM
    from megatron.lite.primitive.kernels.sgl_deepgemm import linear_bf16
    from megatron.lite.primitive.kernels.sgl_norm import visible_input_norm

    seen = {}

    def _hooks(chunk):
        for i, layer in enumerate(chunk.layers):
            def pre(m, inp, i=i):
                seen[("pair", i)] = None if m._residual_pair is None else tuple(t.detach().clone() for t in m._residual_pair)
            layer.attn.qkv.register_forward_pre_hook(pre)
            layer.attn.qkv.register_forward_hook(lambda m, inp, out, i=i: seen.__setitem__(("qkv", i), (out[0] if isinstance(out, tuple) else out).detach().clone()))

    align = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=("rmsnorm", "qkv")), hooks=_hooks)
    n_layers = max(i for (_, i) in seen) + 1
    assert seen[("pair", 0)] is None and all(seen[("pair", i)] is not None for i in range(1, n_layers))
    assert torch.isfinite(align["loss"]).all()
    assert torch.isfinite(align["loss"]).all() and all(torch.isfinite(g).all() for g in align["grads"].values())
    rel = ((lite["loss"] - align["loss"]).abs() / lite["loss"].abs()).item()
    assert rel < 1e-2, rel
    print(f"\nqkv switch: loss rel diff vs lite {rel:.2e} (bitwise equal: {torch.equal(lite['loss'], align['loss'])})", flush=True)


DENSE_SWITCHES = ("rmsnorm", "qkv", "o_proj", "dense_mlp")


def test_all_dense_switches_run_close_to_lite(magi_cfg, magi_source, dist):
    """rmsnorm + qkv + o_proj + dense_mlp together: the whole dense-layer forward (L0-2 analogue) on SGLang bytes."""
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.align.dense_mlp import SglM3DenseMLP
    from megatron.lite.model.minimax_m3.lite import protocol as L
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module

    _magi_deps()
    try:
        deep_gemm_module()
    except ImportError as exc:
        pytest.skip(str(exc))
    seq_lens = BATCHES["packed"]
    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **{**L.MINIMAX_TRAINING_IMPL, "magi_dense_kernel_backend": _dense_backend()}), pre_roll=True)
    torch.manual_seed(7)
    bundle = A.build_model(magi_cfg, impl_cfg=_impl_cfg(A, align_modules=DENSE_SWITCHES))
    chunk = bundle.chunks[0]
    n_dense = sum(1 for layer in chunk.layers if layer.moe is None)
    assert sum(isinstance(m, SglM3DenseMLP) for m in chunk.modules()) == n_dense > 0
    del bundle, chunk
    torch.cuda.empty_cache()
    align = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=DENSE_SWITCHES))
    assert torch.isfinite(align["loss"]).all() and all(torch.isfinite(g).all() for g in align["grads"].values())
    rel = ((lite["loss"] - align["loss"]).abs() / lite["loss"].abs()).item()
    assert rel < 1e-2, rel
    print(f"\nall dense switches {DENSE_SWITCHES}: loss rel diff vs lite {rel:.2e}", flush=True)


ALL_SWITCHES = ("rmsnorm", "router", "o_proj", "qkv", "dense_mlp", "lm_head", "experts")


def test_all_switches_run_and_report_token_logprob_delta(magi_cfg, magi_source, dist):
    """Every switch on: the C5 quantity ``log_probs`` comes from the BI log_softmax of DeepGEMM logits."""
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.align.lm_head import SglVocabParallelOutput
    from megatron.lite.model.minimax_m3.lite import protocol as L
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module

    _magi_deps()
    try:
        deep_gemm_module()
    except ImportError as exc:
        pytest.skip(str(exc))
    seq_lens = BATCHES["packed"]
    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **{**L.MINIMAX_TRAINING_IMPL, "magi_dense_kernel_backend": _dense_backend()}), pre_roll=True)
    torch.manual_seed(7)
    bundle = A.build_model(magi_cfg, impl_cfg=_impl_cfg(A, align_modules=ALL_SWITCHES))
    assert isinstance(bundle.chunks[0].head, SglVocabParallelOutput)
    del bundle
    torch.cuda.empty_cache()
    align = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=ALL_SWITCHES))
    assert torch.isfinite(align["loss"]).all() and all(torch.isfinite(g).all() for g in align["grads"].values())
    rel = ((lite["loss"] - align["loss"]).abs() / lite["loss"].abs()).item()
    assert rel < 1e-2, rel
    lp_l, lp_a = lite["log_probs"], align["log_probs"]
    print(f"\nall switches {ALL_SWITCHES}: loss rel diff vs lite {rel:.2e}; token log_probs differ on "
          f"{(lp_l != lp_a).float().mean().item():.4f} of tokens, max |d| {(lp_l - lp_a).abs().max().item():.2e}", flush=True)


def test_recompute_full_with_switches_keeps_every_gradient_and_the_forward_bytes(magi_cfg, magi_source, dist):
    """recompute=["full"] re-runs the layer forward inside a checkpoint: all switches must still give every parameter a
    gradient (the final norm/head gradient flows through h) and the loss must be bitwise the non-recompute run."""
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module

    _magi_deps()
    try:
        deep_gemm_module()
    except ImportError as exc:
        pytest.skip(str(exc))
    seq_lens = BATCHES["packed"]
    plain = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=ALL_SWITCHES))
    rec = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=ALL_SWITCHES, recompute=["full"]))
    assert torch.equal(plain["loss"], rec["loss"]) and torch.equal(plain["log_probs"], rec["log_probs"])
    from megatron.lite.primitive import recompute as R

    assert not R.is_recomputing()  # the replay scope is balanced
    assert rec["grads"].keys() == plain["grads"].keys(), set(plain["grads"]) ^ set(rec["grads"])
    assert any(k.startswith("layers.0.") for k in rec["grads"]) and any(k.startswith("embed") for k in rec["grads"])
    print(f"\nrecompute=full + all switches: loss/log_probs bitwise equal, {len(rec['grads'])} grads present", flush=True)


@pytest.mark.parametrize("shared", ["separate", "fused"])
def test_experts_switch_runs_close_to_lite_and_shared_modes_differ(magi_cfg, magi_source, dist, shared):
    """align_modules=("router", "experts"): MoE layers show the SGLang Triton fused-MoE bytes; the shared-expert mode is a
    visible contract choice (fused slot vs separate add) so the two modes must not coincide."""
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.align.moe import SglM3MoELayer
    from megatron.lite.model.minimax_m3.lite import protocol as L
    from megatron.lite.primitive.kernels.sgl_deepgemm import deep_gemm_module

    _magi_deps()
    try:
        deep_gemm_module()
    except ImportError as exc:
        pytest.skip(str(exc))
    seq_lens = BATCHES["packed"]
    moe_io = {}

    moe_mods = {}

    def _moe_hooks(tag):
        def install(chunk):
            for i, layer in enumerate(chunk.layers):
                if layer.moe is not None:
                    layer.moe.register_forward_hook(
                        lambda m, inp, out, i=i: moe_io.__setitem__((tag, i), (inp[0].detach().clone(), out.detach().clone())))
                    if tag == "align":
                        moe_mods[i] = layer.moe  # kept alive by this dict for the fp32 reference below
                        layer.moe.router.register_forward_hook(
                            lambda m, inp, out: setattr(moe_mods[[k for k, v in moe_mods.items() if v.router is m][0]], "_routing_seen",
                                                        (out[0].detach().clone(), out[1].detach().clone())))
        return install

    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **{**L.MINIMAX_TRAINING_IMPL, "magi_dense_kernel_backend": _dense_backend()}),
                pre_roll=True, hooks=_moe_hooks("lite"))
    torch.manual_seed(7)
    bundle = A.build_model(magi_cfg, impl_cfg=_impl_cfg(A, align_modules=("router", "experts"), align_moe_shared=shared))
    chunk = bundle.chunks[0]
    n_moe = sum(1 for layer in chunk.layers if layer.moe is not None)
    assert sum(isinstance(m, SglM3MoELayer) for m in chunk.modules()) == n_moe > 0
    del bundle, chunk
    torch.cuda.empty_cache()
    align = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=("router", "experts"), align_moe_shared=shared),
                 hooks=_moe_hooks("align"))
    other = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=("router", "experts"),
                                                               align_moe_shared="fused" if shared == "separate" else "separate"))
    first = min(i for (t, i) in moe_io if t == "lite")
    x_l, y_l = moe_io[("lite", first)]
    x_a, y_a = moe_io[("align", first)]
    assert torch.equal(x_l, x_a), "first MoE layer must see identical inputs (dense layers are lite-identical here)"
    scale = y_l.float().abs().max().item()
    torch.testing.assert_close(y_a.float(), y_l.float(), rtol=2 ** -5, atol=2 ** -6 * scale)  # same math, kernel-level rounding only
    # fp32 reference of the SGLang semantics from the captured routing and the layer's own weights
    moe = moe_mods[first]
    scores, ids = moe._routing_seen
    x2 = x_a.reshape(-1, x_a.shape[-1]).float()
    cfg = moe.shared_expert.config

    def _swiglu(gu):
        gate, up = gu.chunk(2, dim=-1)
        gate = gate.clamp(max=cfg.swiglu_limit)
        return gate * torch.sigmoid(gate * cfg.swiglu_alpha) * (up.clamp(-cfg.swiglu_limit, cfg.swiglu_limit) + 1)

    ref = torch.zeros_like(x2)
    for k in range(ids.shape[1]):
        for e in ids[:, k].unique().tolist():
            rows = ids[:, k] == e
            w13 = getattr(moe.experts.fc1, f"weight{e}").float()
            w2 = getattr(moe.experts.fc2, f"weight{e}").float()
            ref[rows] += (_swiglu(x2[rows] @ w13.t()) @ w2.t()) * scores[rows, k].float()[:, None]
    ref += _swiglu(x2 @ moe.shared_expert.gate_up.linear.weight.float().t()) @ moe.shared_expert.down.linear.weight.float().t()
    torch.testing.assert_close(y_a.reshape(-1, y_a.shape[-1]).float(), ref, rtol=2 ** -5, atol=2 ** -6 * scale)
    print(f"\nfirst MoE layer ({first}) on identical input: SGLang fused-MoE path vs lite dispatcher+TE grouped GEMM: "
          f"{(y_l != y_a).float().mean().item():.4f} of bf16 outputs differ, max |d| {(y_l.float() - y_a.float()).abs().max().item():.3e}; "
          f"fp32 reference max |d| {(y_a.reshape(-1, y_a.shape[-1]).float() - ref).abs().max().item():.3e}", flush=True)
    assert torch.isfinite(align["loss"]).all() and all(torch.isfinite(g).all() for g in align["grads"].values())
    rel = ((lite["loss"] - align["loss"]).abs() / lite["loss"].abs()).item()
    assert rel < 1e-2, rel
    assert not torch.equal(align["log_probs"], other["log_probs"]), "shared-expert modes should differ in bytes"
    print(f"\nexperts switch (shared={shared}): {n_moe} MoE layers swapped, loss rel diff vs lite {rel:.2e}, "
          f"log_probs vs lite differ on {(lite['log_probs'] != align['log_probs']).float().mean().item():.3f} of tokens", flush=True)
