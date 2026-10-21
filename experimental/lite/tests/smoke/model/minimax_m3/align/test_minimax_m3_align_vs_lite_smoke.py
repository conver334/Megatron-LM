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
    noise = {n: (lite["grads"][n].float() - lite2["grads"][n].float()).abs().max().item() for n in lite["grads"]}
    stable = [n for n, v in noise.items() if v == 0.0]
    assert stable, "no run-to-run reproducible grads; A/A control is vacuous"
    bad = [n for n in stable if not torch.equal(lite["grads"][n], align["grads"][n])]
    assert bad == [], bad[:10]
    over = {n: ((lite["grads"][n].float() - align["grads"][n].float()).abs().max().item(), noise[n])
            for n in lite["grads"] if n not in stable}
    over = {n: v for n, v in over.items() if v[0] > 2 * v[1]}  # one A/A sample: allow 2x its spread
    assert over == {}, dict(list(over.items())[:10])
    print(f"\n{tag}: {len(stable)}/{len(noise)} grads bitwise (rest within A/A noise), "
          f"deterministic={_deterministic_supported()}", flush=True)
