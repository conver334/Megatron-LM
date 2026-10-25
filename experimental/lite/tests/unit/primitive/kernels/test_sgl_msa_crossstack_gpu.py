# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax sparse attention, training vs inference on identical inputs: Magi's ``msa_v1`` kernels (indexer GEMM +
CuTe top-k, sparse attention) against the SGLang fork's prefill path run by path (CuTe ``inference.msa_v1`` indexer
GEMM + CUDA radix top-k, gathered and paged sparse attention over the shared ``msa_v1`` kernel). Top-k sets and
attention bytes must agree, including exact score ties, packed documents and long sequences."""

from __future__ import annotations

import importlib

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]

HQ, HKV, D, HI, K, BLK = 64, 4, 128, 4, 16, 128
SCALE = D ** -0.5


def _case(lens, seed, tie_blocks=None):
    torch.manual_seed(seed)
    T = sum(lens)
    q = (torch.randn(T, HQ, D, device="cuda") * 0.5).to(torch.bfloat16)
    k = (torch.randn(T, HKV, D, device="cuda") * 0.5).to(torch.bfloat16)
    v = (torch.randn(T, HKV, D, device="cuda") * 0.5).to(torch.bfloat16)
    iq = (torch.randn(T, HI, D, device="cuda") * 0.5).to(torch.bfloat16)
    ik = (torch.randn(T, 1, D, device="cuda") * 0.5).to(torch.bfloat16)
    if tie_blocks:  # identical key blocks -> exactly tied fp32 block scores
        for d in tie_blocks[1:]:
            ik[d * BLK:(d + 1) * BLK] = ik[tie_blocks[0] * BLK:(tie_blocks[0] + 1) * BLK]
    cu = torch.tensor([0] + torch.cumsum(torch.tensor(lens), 0).tolist(), device="cuda", dtype=torch.int32)
    return q, k, v, iq, ik, cu, torch.tensor(lens, device="cuda", dtype=torch.int32)


def _training(q, k, v, iq, ik, cu, lens, total_pages):
    """The training-side spelling (what Magi's MSAv1 backend calls), kept independent of sgl_msa_prefill."""
    tr_idx, tr_attn = importlib.import_module("msa_v1.indexer"), importlib.import_module("msa_v1.attention")
    T, max_len = int(cu[-1]), int(lens.max())
    topk, _ = tr_idx.forward(iq, ik, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max_len, max_seqlen_kv=max_len, deterministic=True)
    meta = tr_attn.prepare(topk, cu, cu, total_k=T, total_rows=total_pages, max_seqlen_q=max_len, max_seqlen_k=max_len,
                           fragment_indices=None, prepare_kl_schedule=False)
    out, _ = tr_attn.forward(q, k, v, meta, softmax_scale=SCALE, return_softmax_lse=True, partial_dtype=torch.bfloat16)
    return topk, out


def _fork(ci, cw, q, k, v, iq, ik, cu, lens):
    from msa_xstack import fork_prefill

    return fork_prefill(ci, cw, q, k, v, iq, ik, lens.tolist())


def _sets(t):
    return torch.sort(t.to(torch.int64), dim=-1).values


CASES = {
    "single_4096": ([4096], 0, None),
    "long_32768": ([32768], 1, None),
    "packed_1500_2596": ([1500, 2596], 2, None),
    "packed_4_docs_incl_1_token": ([700, 4096, 1, 2048], 3, None),
    "ties_3_way": ([4096], 4, (3, 7, 11)),
    "ties_16_way": ([4096], 5, tuple(range(16))),
    "ties_20_way_long": ([32768], 6, tuple(range(5, 25))),
}


@pytest.mark.parametrize("name", list(CASES))
def test_msa_prefill_training_equals_fork(name):
    from sglang_fork import load_fork_msa

    ci, cw = load_fork_msa()
    lens, seed, ties = CASES[name]
    q, k, v, iq, ik, cu, lens_t = _case(lens, seed, ties)
    fk_topk, gathered, paged, _, total_pages = _fork(ci, cw, q, k, v, iq, ik, cu, lens_t)
    tr_topk, tr_out = _training(q, k, v, iq, ik, cu, lens_t, total_pages)
    assert tr_topk.shape == fk_topk.shape == (HI, int(cu[-1]), K)
    same = (_sets(tr_topk) == _sets(fk_topk)).all(dim=-1)
    assert same.all(), f"top-k sets differ on {(~same).float().mean().item():.4f} of (head, query) rows"
    assert torch.equal(tr_out, gathered), f"training attention vs fork gathered: {(tr_out != gathered).float().mean().item():.6f} differ"
    assert torch.equal(gathered, paged), f"fork gathered vs paged: {(gathered != paged).float().mean().item():.6f} differ"


def test_msa_fork_path_is_deterministic_and_order_only():
    """Two runs agree bitwise (scores on the written, causal entries); the attention bytes do not depend on the
    top-k order handed to the CSR builder."""
    from sglang_fork import load_fork_msa

    ci, cw = load_fork_msa()
    q, k, v, iq, ik, cu, lens = _case([4096, 2048], 9)
    a = _fork(ci, cw, q, k, v, iq, ik, cu, lens)
    b = _fork(ci, cw, q, k, v, iq, ik, cu, lens)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    # fp32 block scores: the kernel writes only the causal (past) blocks of each query, the rest of the buffer is uninitialised
    pos = torch.cat([torch.arange(int(l), device="cuda") for l in lens.tolist()])
    valid = (torch.arange(a[3].shape[-1], device="cuda")[None, :] <= (pos // BLK)[:, None])[None]
    assert torch.equal(torch.where(valid, a[3], 0.0), torch.where(valid, b[3], 0.0))
    max_len = int(lens.max())
    perm = torch.argsort(torch.rand(a[0].shape, device="cuda"), dim=-1)
    shuffled = torch.gather(a[0], -1, perm)
    out_shuffled, _, _ = cw.cute_topk_sparse_attention(q, k, v, shuffled, cu, cu, max_len, max_len, BLK, K, SCALE)
    assert torch.equal(out_shuffled, a[1])


@pytest.mark.parametrize("name", ["packed_1500_2596", "ties_16_way", "long_32768"])
def test_sgl_msa_prefill_equals_fork_chain(name):
    """The training-side module (training kernels in the inference ABI) == the fork's paged chain by path."""
    from megatron.lite.primitive.kernels.sgl_msa import sgl_msa_prefill
    from sglang_fork import load_fork_msa

    ci, cw = load_fork_msa()
    lens, seed, ties = CASES[name]
    q, k, v, iq, ik, cu, lens_t = _case(lens, seed, ties)
    _, _, paged, _, _ = _fork(ci, cw, q, k, v, iq, ik, cu, lens_t)
    ours = sgl_msa_prefill(q, k, v, iq, ik, cu.tolist())
    assert torch.equal(ours, paged)


def test_kl_recipe_abi_changes_the_attention_bytes():
    """Datum: Magi under the MiniMax KL recipe calls the same kernel with lse_temperature_scale=2 and the temperature
    LSE, which selects the normalised-partial-statistics path; inference (lse 1, raw statistics) differs."""
    pytest.importorskip("msa_v1.attention")
    tr_attn = importlib.import_module("msa_v1.attention")
    from megatron.lite.primitive.kernels.sgl_msa import sgl_msa_prefill

    q, k, v, iq, ik, cu, lens = _case([1500, 2596], 2)
    infer = sgl_msa_prefill(q, k, v, iq, ik, cu.tolist())
    tr_idx = importlib.import_module("msa_v1.indexer")
    max_len = int(lens.max())
    topk, _ = tr_idx.forward(iq, ik, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max_len, max_seqlen_kv=max_len,
                             lse_temperature=2.0, deterministic=True)  # scores scaled by 1/T: same ranking
    pages = sum((l + BLK - 1) // BLK for l in lens.tolist())
    meta = tr_attn.prepare(topk, cu, cu, total_k=int(cu[-1]), total_rows=pages, max_seqlen_q=max_len, max_seqlen_k=max_len,
                           fragment_indices=None, prepare_kl_schedule=True)
    out_kl, _, _ = tr_attn.forward(q, k, v, meta, softmax_scale=SCALE, return_softmax_lse=True, lse_temperature_scale=2.0,
                                   return_temperature_lse=True, partial_dtype=torch.bfloat16)
    frac = (out_kl != infer).float().mean().item()
    print(f"\nKL-recipe ABI (lse_temperature_scale 2, temperature LSE) vs inference ABI: {frac:.4f} of bytes differ, "
          f"max |d| {(out_kl.float() - infer.float()).abs().max().item():.3e}")
    assert frac > 0  # the msa_core switch exists for this


@pytest.mark.parametrize("name", ["ties_warps_long", "ties_large_radix"])
def test_exact_ties_across_warps_on_the_radix_path(name):
    """Ties placed beyond warp 0 on the radix top-k path (129..4096 visible blocks): the fork resolves them by
    shared-memory atomicAdd arrival order, the training top-k by the lowest block id. Measures whether the selections
    stay equal and run-to-run stable; a mismatch here is a tie-rule datum, not a kernel defect."""
    from sglang_fork import load_fork_msa

    ci, cw = load_fork_msa()
    lens, seed, ties = {"ties_warps_long": ([32768], 7, tuple(range(20, 46))), "ties_large_radix": ([70000], 8, tuple(range(513, 540)))}[name]
    q, k, v, iq, ik, cu, lens_t = _case(lens, seed, ties)
    runs = [_fork(ci, cw, q, k, v, iq, ik, cu, lens_t)[0] for _ in range(3)]
    stable = all(torch.equal(_sets(runs[0]), _sets(r)) for r in runs[1:])
    tr_topk, _ = _training(q, k, v, iq, ik, cu, lens_t, sum((l + BLK - 1) // BLK for l in lens))
    same = (_sets(tr_topk) == _sets(runs[0])).all(dim=-1)
    print(f"\n[{name}] fork radix top-k run-to-run stable: {stable}; set equality vs training: {same.float().mean().item():.6f}")
    assert stable and same.all(), "exact-tie selection differs: record as a tie-rule contract item"


def test_chunked_prefill_extend_equals_full_prefill():
    """The server scores a cached prefix + new chunk (prefix_lens > 0) with the same chain: the new chunk's rows must
    equal the full-prefill rows (chunked prefill == full prefill), and the full prefill == sgl_msa_prefill."""
    from megatron.lite.primitive.kernels.sgl_msa import sgl_msa_prefill
    from msa_xstack import fork_prefill
    from sglang_fork import load_fork_msa

    ci, cw = load_fork_msa()
    q, k, v, iq, ik, cu, lens = _case([4096], 10)
    _, _, full, _, _ = fork_prefill(ci, cw, q, k, v, iq, ik, [4096])
    assert torch.equal(full, sgl_msa_prefill(q, k, v, iq, ik, [0, 4096]))
    for split in (2048, 1500):
        _, _, chunk2, _, _ = fork_prefill(ci, cw, q[split:].contiguous(), k, v, iq[split:].contiguous(), ik, [4096 - split], prefix_lens=[split])
        frac = (chunk2 != full[split:]).float().mean().item()
        assert frac == 0.0, f"chunked prefill (prefix {split}) differs from full prefill on {frac:.6f} of bytes"


def test_attention_tp_shapes_leave_the_raw_statistics_path_datum():
    """Datum: with attention TP=4 the server runs the same CuTe kernels on 16 q / 1 kv heads, where the kernel keeps
    normalised partial statistics instead of raw max/sum; the bytes of those heads differ from the TP=1 run."""
    from msa_xstack import fork_prefill
    from sglang_fork import load_fork_msa

    ci, cw = load_fork_msa()
    q, k, v, iq, ik, cu, lens = _case([4096], 11)
    topk, _, full, _, _ = fork_prefill(ci, cw, q, k, v, iq, ik, [4096])
    # rank 0 of attention TP=4: q heads 0..15, kv head 0, index head 0 (its top-k is the same rows of the 4-head run)
    q0, k0, v0 = q[:, :16].contiguous(), k[:, :1].contiguous(), v[:, :1].contiguous()
    cu_t = torch.tensor([0, 4096], device="cuda", dtype=torch.int32)
    out_tp4, _, _ = cw.cute_topk_sparse_attention(q0, k0, v0, topk[:1].contiguous(), cu_t, cu_t, 4096, 4096, BLK, K, SCALE)
    frac = (out_tp4 != full[:, :16]).float().mean().item()
    print(f"\nattention TP=4 shape (16 q / 1 kv heads) vs TP=1 bytes of the same heads: {frac:.4f} differ")
    assert frac > 0, "normalised-statistics path produced the raw-statistics bytes; re-check the kernel gate"


def test_abi_controls_pin_the_normalised_statistics_gate():
    """The raw-vs-normalised gate is `return_temperature_lse and lse_temperature_scale != 1`: either flag alone keeps
    the inference bytes; both together (the KL recipe) change them."""
    pytest.importorskip("msa_v1.attention")
    from megatron.lite.primitive.kernels.sgl_msa import sgl_msa_prefill

    tr_idx, tr_attn = importlib.import_module("msa_v1.indexer"), importlib.import_module("msa_v1.attention")
    q, k, v, iq, ik, cu, lens = _case([1500, 2596], 2)
    infer = sgl_msa_prefill(q, k, v, iq, ik, cu.tolist())
    max_len = int(lens.max())
    pages = sum((l + BLK - 1) // BLK for l in lens.tolist())
    topk, _ = tr_idx.forward(iq, ik, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max_len, max_seqlen_kv=max_len, deterministic=True)
    meta = tr_attn.prepare(topk, cu, cu, total_k=int(cu[-1]), total_rows=pages, max_seqlen_q=max_len, max_seqlen_k=max_len,
                           fragment_indices=None, prepare_kl_schedule=True)
    scale_only, _ = tr_attn.forward(q, k, v, meta, softmax_scale=SCALE, return_softmax_lse=True, lse_temperature_scale=2.0,
                                    return_temperature_lse=False, partial_dtype=torch.bfloat16)
    lse_only = tr_attn.forward(q, k, v, meta, softmax_scale=SCALE, return_softmax_lse=True, lse_temperature_scale=1.0,
                               return_temperature_lse=True, partial_dtype=torch.bfloat16)[0]
    both = tr_attn.forward(q, k, v, meta, softmax_scale=SCALE, return_softmax_lse=True, lse_temperature_scale=2.0,
                           return_temperature_lse=True, partial_dtype=torch.bfloat16)[0]
    assert torch.equal(scale_only, infer) and torch.equal(lse_only, infer)
    assert not torch.equal(both, infer)
