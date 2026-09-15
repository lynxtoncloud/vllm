# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    _use_rocm_sparse_triton,
    fit_kpool_indices_to_aiter,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _sparse_kv_row_offset,
    _validate_dsv4_sparse_dims,
    _validate_sparse_dims,
)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.parametrize("heads,dim", [(1, 64), (16, 128), (64, 128)])
def test_batched_prefill_logits_preserve_mask_and_replay(heads, dim):
    """Graph replay must consume new queries and GPU-only ragged boundaries."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch
    from vllm.v1.attention.ops.rocm_mqa_logits import prefill_mqa_logits

    torch.manual_seed(21)
    fp8 = current_platform.fp8_dtype()
    q = torch.randn(3, heads, dim, device="cuda").to(fp8)
    k = torch.randn(137, dim, device="cuda").to(fp8)
    scale = torch.rand(137, 1, device="cuda") * 0.01
    weights = torch.rand(3, heads, device="cuda")
    starts = torch.tensor([0, 7, 129], device="cuda", dtype=torch.int32)
    ends = torch.tensor([0, 133, 137], device="cuda", dtype=torch.int32)
    for _ in range(2):
        prefill_mqa_logits(q, (k, scale), weights, starts, ends)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = prefill_mqa_logits(q, (k, scale), weights, starts, ends)
    for step in range(3):
        q.copy_(torch.randn(q.shape, device="cuda").to(fp8))
        ends[0] = step * 31
        graph.replay()
        expected = fp8_mqa_logits_torch(q, (k, scale), weights, starts, ends)
        torch.testing.assert_close(output, expected, atol=0.005, rtol=0.005)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.parametrize("page", [1, 16, 64])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("two_dimensional_lengths", [False, True])
@pytest.mark.parametrize("padded_pages", [False, True])
def test_batched_paged_logits_replay_reads_updated_pages_and_lengths(
    page, next_n, two_dimensional_lengths, padded_pages
):
    """Check AMD tile layout, scale offsets, partial pages and speculative rows."""
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_paged_mqa_logits_torch
    from vllm.v1.attention.ops.rocm_mqa_logits import paged_mqa_logits

    torch.manual_seed(22)
    fp8 = current_platform.fp8_dtype()
    batch, heads, dim, pages = 2, 16, 128, 5
    q = torch.randn(batch, next_n, heads, dim, device="cuda").to(fp8)
    values = torch.randn(pages, page, dim, device="cuda").to(fp8)
    if page > 1:
        values = values.reshape(pages, page // 16, 16, dim // 16, 16)
        values = values.transpose(2, 3).contiguous()
    scales = torch.rand(pages, page, device="cuda") * 0.01
    packed = torch.cat(
        (values.reshape(pages, -1).view(torch.uint8), scales.view(torch.uint8)), dim=1
    ).reshape(pages, page, 1, dim + 4)
    if padded_pages:
        storage = torch.empty(
            pages, page * (dim + 4) + 16, dtype=torch.uint8, device="cuda"
        )
        view = storage[:, : page * (dim + 4)].view_as(packed)
        view.copy_(packed)
        packed = view
    weights = torch.rand(batch * next_n, heads, device="cuda")
    table = torch.tensor([[3, 1, 4], [4, 2, 0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([page * 3 - 1, 0], device="cuda", dtype=torch.int32)
    if two_dimensional_lengths:
        lengths = (
            (lengths[:, None] - next_n + 1 + torch.arange(next_n, device="cuda"))
            .clamp_min(0)
            .int()
        )
    limit = page * 3 + 11  # Output bound can exceed the current block table width.
    for _ in range(2):
        paged_mqa_logits(q, packed, weights, lengths, table, limit)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = paged_mqa_logits(q, packed, weights, lengths, table, limit)
    for step in range(3):
        table.copy_(table.flip(1))
        q.copy_(torch.randn(q.shape, device="cuda").to(fp8))
        lengths[1] = min(page * 3, next_n + step)
        graph.replay()
        expected = fp8_paged_mqa_logits_torch(q, packed, weights, lengths, table, limit)
        torch.testing.assert_close(output, expected, atol=0.001, rtol=0.001)


@triton.jit
def _store_sparse_kv_row_offset_kernel(slot_ptr, output_ptr, stride: tl.constexpr):
    slot = tl.load(slot_ptr)
    tl.store(output_ptr, _sparse_kv_row_offset(slot, stride))


def test_fit_kpool_indices_preserves_tail_and_best_history():
    token_indices = torch.tensor(
        [
            [10, 9, 8, 7, 6, 5, 100, 101],
            [10, 9, 8, -1, -1, -1, 100, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=6)

    assert fitted.tolist() == [
        [10, 9, 8, 7, 100, 101],
        [10, 9, 8, 100, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ]


def test_fit_kpool_indices_exact_width_is_noop():
    token_indices = torch.tensor([[3, 2, 1, -1]], dtype=torch.int32)

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=4)

    assert fitted.data_ptr() == token_indices.data_ptr()


def test_fit_kpool_indices_rejects_narrow_input():
    with pytest.raises(ValueError, match="at least topk_tokens"):
        fit_kpool_indices_to_aiter(
            torch.zeros((1, 3), dtype=torch.int32), topk_tokens=4
        )


@pytest.mark.parametrize(
    (
        "kv_cache_dtype",
        "head_size",
        "num_prefills",
        "num_decodes",
        "num_decode_tokens",
        "max_query_len",
        "expected",
    ),
    [
        ("auto", 512, 1, 0, 0, 32, True),
        ("auto", 512, 1, 2, 2, 32, True),
        ("auto", 512, 0, 2, 2, 1, True),
        ("fp8", 512, 1, 0, 0, 32, False),
        ("auto", 576, 1, 0, 0, 32, False),
        ("auto", 512, 0, 2, 4, 2, True),
        ("auto", 512, 0, 2, 12, 6, True),
        ("auto", 512, 0, 0, 0, 0, False),
    ],
)
def test_rocm_sparse_triton_route(
    kv_cache_dtype,
    head_size,
    num_prefills,
    num_decodes,
    num_decode_tokens,
    max_query_len,
    expected,
):
    """Validate Triton routing for prefill, decode, and MTP verification."""
    assert (
        _use_rocm_sparse_triton(
            kv_cache_dtype=kv_cache_dtype,
            head_size=head_size,
            kv_lora_rank=512,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            max_query_len=max_query_len,
        )
        is expected
    )


@pytest.mark.parametrize("num_heads", [8, 12])
def test_rocm_sparse_triton_route_preserves_padded_sinks(monkeypatch, num_heads):
    captured = {}

    def fake_rocm_sparse_attn_prefill(**kwargs):
        output = kwargs["output"]
        captured["attn_sink"] = kwargs["attn_sink"]
        output.copy_(
            captured["attn_sink"].to(output.dtype).view(1, -1, 1).expand_as(output)
        )

    monkeypatch.setattr(
        sparse_mod, "rocm_sparse_attn_prefill", fake_rocm_sparse_attn_prefill
    )

    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = 512
    impl.kv_cache_dtype = "auto"
    impl.scale = 512**-0.5
    impl.sinks = torch.arange(num_heads, dtype=torch.float32)

    q = torch.zeros(2, 16, 512, dtype=torch.bfloat16)
    kv = torch.zeros(4, 1, 512, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        max_query_len=2,
        paged_kv_indices=torch.empty(0, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(3, dtype=torch.int32),
    )

    output, lse = impl._forward_mla(SimpleNamespace(), q, kv, metadata)

    if num_heads == 8:
        expected_sinks = impl.sinks.repeat_interleave(2)
    else:
        expected_sinks = torch.cat((impl.sinks, impl.sinks[:4]))
    torch.testing.assert_close(captured["attn_sink"], expected_sinks)
    assert output.shape == (2, num_heads, 512)
    torch.testing.assert_close(
        output[:, :, 0].float(),
        impl.sinks.expand(2, -1),
    )
    assert lse is None


def test_rocm_sparse_attention_accepts_glm_nope_dimensions():
    _validate_sparse_dims(512, 512, 0, "test")


def test_rocm_sparse_attention_rejects_inconsistent_dimensions():
    with pytest.raises(AssertionError, match="expected head_dim"):
        _validate_sparse_dims(511, 512, 0, "test")


def test_dsv4_sparse_attention_keeps_layout_constraint():
    _validate_dsv4_sparse_dims(512, 448, 64, "test")
    with pytest.raises(AssertionError, match="expects 448 NoPE dims"):
        _validate_dsv4_sparse_dims(512, 512, 0, "test")


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_sparse_prefill_kv_row_offset_does_not_overflow_int32():
    # GLM's 640-token pages cross the signed-int32 address boundary at block
    # 6554 for a 512-element KV row. The production kernel must promote the
    # slot before multiplying by the row stride.
    slot = torch.tensor([6554 * 640], dtype=torch.int32, device="cuda")
    output = torch.empty(1, dtype=torch.int64, device="cuda")

    _store_sparse_kv_row_offset_kernel[(1,)](slot, output, stride=512)

    assert output.item() == 6554 * 640 * 512
