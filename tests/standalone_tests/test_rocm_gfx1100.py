# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the eager fallbacks, without loading native extensions.

Run with --confcutdir=tests/standalone_tests. These tests intentionally avoid
the model/GPU fixtures in tests/conftest.py. They also run unchanged on ROCm
with GFX1100_TEST_DEVICE=cuda for numerical GPU validation.
"""

import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.utils import rocm_gfx1100 as fallback

DEVICE = os.getenv("GFX1100_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("ignore", [False, True])
def test_align_preserves_original_routes_with_nonidentity_ep_map(ignore):
    routes = torch.tensor([[287, 0], [2, 287], [0, 2]], device=DEVICE)
    mapping = torch.full((288,), -1, dtype=torch.int32, device=DEVICE)
    mapping[2], mapping[287] = 1, 0
    s, e, n = fallback.moe_align(routes, 4, 288, mapping, False, ignore)
    if ignore:
        expected_s, expected_e = [0, 3, 6, 6, 2, 5, 6, 6], [0, 1]
    else:
        expected_s = [1, 4, 6, 6, 2, 5, 6, 6, 0, 3, 6, 6]
        expected_e = [-1, 1, 0]
    assert n.item() == len(expected_s)
    assert s[: n.item()].tolist() == expected_s
    assert e[: len(expected_e)].tolist() == expected_e
    assert (s[n.item() :] == routes.numel()).all()
    assert (e[len(expected_e) :] == -1).all()


@pytest.mark.parametrize("count", [0, 1, 17])
def test_align_empty_and_cross_block_routes(count):
    routes = torch.full((count, 1), 287, dtype=torch.int32, device=DEVICE)
    s, e, n = fallback.moe_align(routes, 16, 288, pad_sorted_ids=True)
    expected = (count + 15) // 16 * 16
    assert n.item() == expected
    assert s[:count].tolist() == list(range(count))
    assert (s[count:] == count).all()
    assert (e[: expected // 16] == 287).all()


def test_align_filters_invalid_and_nonlocal_routes():
    routes = torch.tensor([[-1, 3, 0, 1]], device=DEVICE)
    mapping = torch.full((3,), -1, dtype=torch.int32, device=DEVICE)
    s, e, n = fallback.moe_align(routes, 4, 3, mapping, False, True)
    assert n.item() == 0
    assert (s == 4).all() and (e == -1).all()


def test_moe_sum_ignores_nan_padding_and_uses_fp32_accumulation():
    x = torch.tensor(
        [[[256.0], [1], [-256], [1], [float("nan")], [float("nan")]]],
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    routes = torch.tensor([[0, 0, 0, 0, -1, 1]], device=DEVICE)
    mapping = torch.tensor([0, -1], device=DEVICE)
    output = torch.empty((1, 1), dtype=x.dtype, device=DEVICE)
    fallback.moe_sum(x, output, routes, mapping)
    assert output.item() == 2


def test_prefill_topk_relative_indices_short_rows_and_output_view():
    logits = torch.arange(40, device=DEVICE).float().repeat(3, 1)
    starts = torch.tensor([5, 10, 20], device=DEVICE)
    ends = torch.tensor([40, 13, 20], device=DEVICE)
    backing = torch.full((3, 12), -99, dtype=torch.int32, device=DEVICE)
    output = backing[:, :8]
    fallback.topk_prefill(logits, starts, ends, output)
    assert output[0].sort().values.tolist() == list(range(27, 35))
    assert output[1].tolist() == [0, 1, 2, -1, -1, -1, -1, -1]
    assert (output[2] == -1).all() and (backing[:, 8:] == -99).all()


@pytest.mark.parametrize(
    "lens,ends", [([20, 2], [19, 20, 1, 2]), ([[19, 20], [0, 3]], [19, 20, 0, 3])]
)
def test_decode_topk_causal_lengths(lens, ends):
    logits = torch.arange(20, device=DEVICE).float().repeat(4, 1)
    output = torch.empty((4, 8), dtype=torch.int32, device=DEVICE)
    fallback.topk_decode(logits, 2, torch.tensor(lens, device=DEVICE), output)
    for row, end in enumerate(ends):
        count = min(8, end)
        assert output[row, :count].sort().values.tolist() == list(
            range(max(0, end - 8), end)
        )
        assert (output[row, count:] == -1).all()


@pytest.mark.parametrize("seq_len", [0, 1, 3, 4, 2047, 2048, 2049, 2051, 4096, 8191])
def test_kpool_compaction_preserves_full_history_and_incomplete_tail(seq_len):
    """Attention must retain up to 2048 history tokens AND the 0..3 live tail."""
    history_count = min(seq_len // 4, 512) * 4
    # Descending history also catches accidental reordering of top-k output.
    history = torch.arange(history_count, device=DEVICE).flip(0)
    tail = torch.arange(seq_len // 4 * 4, seq_len, device=DEVICE)
    indices = torch.full((1, 2051), -1, dtype=torch.int32, device=DEVICE)
    indices[0, :history_count] = history
    indices[0, 2048 : 2048 + tail.numel()] = tail
    original = indices.clone()
    packed = fallback.compact_sparse_indices(indices)
    expected = torch.cat((history, tail)).to(torch.int32)
    torch.testing.assert_close(packed[0, : expected.numel()], expected)
    assert (packed[0, expected.numel() :] == -1).all()
    torch.testing.assert_close(indices, original)
    # The metadata may reserve up to 3 extra positions; all must stay masked.
    ragged_row = packed[0, : min(seq_len, 2051)]
    torch.testing.assert_close(ragged_row[ragged_row >= 0], expected)


@pytest.mark.parametrize("rope", [0, 2])
def test_mla_cache_slots_zero_rope_and_noncontiguous_cache(rope):
    kv = torch.arange(20, device=DEVICE).reshape(5, 4).bfloat16()
    pe = torch.ones((5, rope), device=DEVICE, dtype=kv.dtype)
    backing = torch.full((2, 4, 2 * (4 + rope)), -99, device=DEVICE, dtype=kv.dtype)
    cache = backing[..., ::2]
    slots = torch.tensor([-1, 3, 4, 7], device=DEVICE)
    expected = cache.clone()
    for row, slot in ((1, 3), (2, 4), (3, 7)):
        expected[slot // 4, slot % 4] = torch.cat((kv[row], pe[row]))
    fallback.concat_and_cache_mla(kv, pe, cache, slots, "auto")
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    assert (backing[..., 1::2] == -99).all()
    fallback.concat_and_cache_mla(kv, pe, cache, slots.fill_(-1), "auto")
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


def config():
    return NS(
        model_config=NS(
            hf_text_config=NS(
                model_type="glm5_next_text", qk_rope_head_dim=0, index_kpool=4
            ),
            dtype=torch.bfloat16,
            quantization=None,
            enforce_eager=True,
            max_model_len=8192,
        ),
        scheduler_config=NS(max_num_seqs=1),
        speculative_config=None,
        cache_config=NS(cache_dtype="auto"),
        kernel_config=NS(moe_backend="auto", ir_op_priority=NS()),
        compilation_config=NS(cudagraph_mm_encoder=True),
    )


def test_profile_selects_safe_backends_idempotently():
    cfg = config()
    fallback.configure(cfg)
    fallback.configure(cfg)
    assert cfg.kernel_config.moe_backend == "triton"
    assert cfg.kernel_config.ir_op_priority.rms_norm == ["native"]
    assert cfg.kernel_config.ir_op_priority.fused_add_rms_norm == ["native"]
    assert not cfg.compilation_config.cudagraph_mm_encoder


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("model_config", "enforce_eager", False),
        ("model_config", "quantization", "fp8"),
        ("model_config", "max_model_len", 1048576),
        ("scheduler_config", "max_num_seqs", 512),
        ("cache_config", "cache_dtype", "fp8"),
        ("kernel_config", "moe_backend", "aiter"),
    ],
)
def test_profile_rejects_unvalidated_config(section, field, value):
    cfg = config()
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError):
        fallback.configure(cfg)


def test_profile_rejects_unsupported_pool_before_loading_weights():
    cfg = config()
    cfg.model_config.hf_text_config.index_kpool = 16
    with pytest.raises(ValueError, match="index_kpool=4"):
        fallback.configure(cfg)


def test_disabled_flag_does_not_initialize_gpu(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_GFX1100_GLM53", "0")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: pytest.fail("disabled fallback must not query GPUs"),
    )
    assert not fallback.enabled(torch.device("cuda:0"))
