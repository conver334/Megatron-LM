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


def _impl_cfg(proto, **kw):
    from megatron.lite.runtime.contracts import ParallelConfig

    return proto.ImplConfig(parallel=ParallelConfig(), optimizer=None, deterministic=_deterministic_supported(),
                            magi_chunk_size=CHUNK, **kw)


def _run(proto, cfg, src, seq_lens, impl_cfg, *, pre_roll=False):
    from megatron.lite.model.minimax_m3.align.protocol import roll_packed_labels
    from megatron.lite.runtime.contracts.data import PackedBatch

    torch.manual_seed(7)
    bundle = proto.build_model(cfg, impl_cfg=impl_cfg)
    chunk = bundle.chunks[0]
    proto.load_hf_weights(chunk, src, cfg, bundle.parallel_state)
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
    result = dict(loss=loss.detach().clone(), grads=grads, logits=None if logits is None else logits.detach().clone())
    del bundle, chunk
    torch.cuda.empty_cache()
    return result


@pytest.mark.parametrize("tag", list(BATCHES))
def test_align_without_switches_matches_lite_bitwise(magi_cfg, magi_source, dist, tag):
    from megatron.lite.model.minimax_m3.align import protocol as A
    from megatron.lite.model.minimax_m3.lite import protocol as L

    _magi_deps()
    seq_lens = BATCHES[tag]
    knobs = dict(L.MINIMAX_TRAINING_IMPL)
    lite = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **knobs), pre_roll=True)
    lite2 = _run(L, magi_cfg, magi_source, seq_lens, _impl_cfg(L, **knobs), pre_roll=True)  # A/A control
    align = _run(A, magi_cfg, magi_source, seq_lens, _impl_cfg(A, align_modules=()))
    assert torch.equal(lite["loss"], lite2["loss"]), "lite forward is not reproducible on this image"
    assert torch.equal(lite["loss"], align["loss"]), (lite["loss"].item(), align["loss"].item())
    if lite["logits"] is not None:
        assert torch.equal(lite["logits"], lite2["logits"])
        assert torch.equal(lite["logits"], align["logits"])
    assert lite["grads"].keys() == align["grads"].keys()
    if _deterministic_supported():
        bad = [n for n in lite["grads"] if not torch.equal(lite["grads"][n], align["grads"][n])]
        assert bad == [], bad[:10]
        return
    # non-deterministic backward: lite-vs-align relative spread must sit inside the lite A/A relative spread
    def _rel(a, b):
        return ((a.float() - b.float()).abs().max() / a.float().abs().max().clamp_min(1e-30)).item()

    noise = {n: _rel(lite["grads"][n], lite2["grads"][n]) for n in lite["grads"]}
    diff = {n: _rel(lite["grads"][n], align["grads"][n]) for n in lite["grads"]}
    bound = 2 * max(noise.values())
    assert bound > 0, "lite A/A grads all bitwise but Magi reports no deterministic mode; revisit the criterion"
    over = {n: (diff[n], noise[n]) for n in diff if diff[n] > bound}
    assert over == {}, (bound, dict(list(over.items())[:10]))
    bitwise = sum(1 for n in diff if diff[n] == 0.0)
    print(f"\n{tag}: forward bitwise; grads {bitwise}/{len(diff)} bitwise, max rel diff {max(diff.values()):.2e} "
          f"vs A/A bound {bound:.2e} (Magi deterministic unsupported)", flush=True)
