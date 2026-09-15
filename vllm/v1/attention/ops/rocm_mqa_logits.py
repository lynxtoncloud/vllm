# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched ROCm indexer logits without host reads of sequence metadata.

FP8 values are widened to BF16 before dot products: no FP8 matrix instruction
is required. Each program reduces all indexer heads into one key tile, avoiding
the full [heads, queries, keys] intermediate used by the Torch reference.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _mqa_logits(
    Q,
    K,
    S,
    W,
    LENGTHS,
    STARTS,
    TABLE,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    NEXT: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    LENS_2D: tl.constexpr,
    q_row: tl.constexpr,
    q_head: tl.constexpr,
    q_dim: tl.constexpr,
    k_row: tl.constexpr,
    k_dim: tl.constexpr,
    w_row: tl.constexpr,
    w_head: tl.constexpr,
    lens_row: tl.constexpr,
    lens_col: tl.constexpr,
    table_row: tl.constexpr,
    table_col: tl.constexpr,
    scale_row: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    keys = tl.program_id(1) * BN + tl.arange(0, BN)
    h = tl.arange(0, BH)
    d = tl.arange(0, BD)
    if PAGED:
        batch = row // NEXT
        step = row % NEXT
        end = tl.load(LENGTHS + batch * lens_row + step * lens_col)
        if not LENS_2D:
            end = end - NEXT + 1 + step
        start = 0
    else:
        start = tl.load(STARTS + row * lens_row)
        end = tl.load(LENGTHS + row * lens_col)
    valid = (keys < N) & (keys >= start) & (keys < end)
    if tl.sum(valid.to(tl.int32), axis=0) > 0:
        query = tl.load(
            Q + row.to(tl.int64) * q_row + h[:, None] * q_head + d[None, :] * q_dim,
            (h[:, None] < H) & (d[None, :] < D),
            other=0,
        ).to(tl.bfloat16)
        if PAGED:
            valid = valid & (keys < TABLE_COLS * PAGE)
            page = tl.load(
                TABLE + batch * table_row + (keys // PAGE) * table_col,
                valid,
                other=-1,
            ).to(tl.int64)
            valid = valid & (page >= 0)
            token = keys % PAGE
            if PAGE == 1:
                offset = d[:, None]
            else:
                # AMD indexer cache packs 16-token x 16-channel tiles, then scales.
                offset = (
                    (token[None, :] // 16) * (D // 16) * 256
                    + (d[:, None] // 16) * 256
                    + (token[None, :] % 16) * 16
                    + d[:, None] % 16
                )
            value = tl.load(
                K + page[None, :] * k_row + offset,
                valid[None, :] & (d[:, None] < D),
                other=0,
            ).to(tl.bfloat16)
            scale = tl.load(
                S + page * (k_row // 4) + PAGE * D // 4 + token,
                valid,
                other=0,
            )
        else:
            value = tl.load(
                K + keys[None, :].to(tl.int64) * k_row + d[:, None] * k_dim,
                valid[None, :] & (d[:, None] < D),
                other=0,
            ).to(tl.bfloat16)
            scale = tl.load(S + keys * scale_row, valid, other=0)
        scores = tl.dot(query, value)
        if not PAGED:
            # Match the reference BF16 einsum result before applying the scale.
            scores = scores.to(tl.bfloat16).to(tl.float32) * scale[None, :]
        weights = tl.load(W + row * w_row + h * w_head, h < H, other=0)
        result = tl.sum(tl.maximum(scores, 0.0) * weights[:, None], axis=0)
        if PAGED:
            result = result * scale
    else:
        result = tl.full((BN,), -float("inf"), tl.float32)
    tl.store(
        OUT + row.to(tl.int64) * N + keys,
        tl.where(valid, result, -float("inf")),
        keys < N,
    )


def prefill_mqa_logits(q, kv, weights, starts, ends):
    k, scales = kv
    m, h, d = q.shape
    n = k.shape[0]
    if d % 16 or d > 256 or h > 128:
        raise ValueError("ROCm MQA requires D divisible by 16 and <=256, H<=128")
    output = torch.empty((m, n), dtype=torch.float32, device=q.device)
    if m and n:
        _mqa_logits[(m, triton.cdiv(n, 64))](
            q,
            k,
            scales,
            weights,
            ends,
            starts,
            starts,
            output,
            n,
            h,
            d,
            0,
            1,
            1,
            False,
            False,
            *q.stride(),
            *k.stride(),
            *weights.stride(),
            starts.stride(0),
            ends.stride(0),
            0,
            0,
            scales.stride(0),
            BH=max(16, triton.next_power_of_2(h)),
            BD=triton.next_power_of_2(d),
            BN=64,
            num_warps=4,
        )
    return output


def paged_mqa_logits(q, cache, weights, lengths, table, max_model_len):
    batch, next_n, h, d = q.shape
    page = cache.shape[1]
    if d % 16 or d > 256 or h > 128 or (page != 1 and page % 16):
        raise ValueError("Unsupported ROCm paged MQA dimensions")
    if (
        cache.stride(-1) != 1
        or cache.stride(1) != d + 4
        or cache.stride(0) % 4
        or q.stride(0) != next_n * q.stride(1)
    ):
        raise ValueError("Paged MQA requires contiguous payloads and query rows")
    output = torch.empty(
        (batch * next_n, max_model_len), dtype=torch.float32, device=q.device
    )
    if batch and max_model_len:
        _mqa_logits[(batch * next_n, triton.cdiv(max_model_len, 64))](
            q,
            cache.view(current_platform.fp8_dtype()),
            cache.view(torch.float32),
            weights,
            lengths,
            lengths,
            table,
            output,
            max_model_len,
            h,
            d,
            table.shape[1],
            next_n,
            page,
            True,
            lengths.ndim == 2,
            q.stride(1),
            q.stride(2),
            q.stride(3),
            cache.stride(0),
            1,
            *weights.stride(),
            lengths.stride(0),
            lengths.stride(1) if lengths.ndim == 2 else 0,
            *table.stride(),
            0,
            BH=max(16, triton.next_power_of_2(h)),
            BD=triton.next_power_of_2(d),
            BN=64,
            num_warps=4,
        )
    return output
