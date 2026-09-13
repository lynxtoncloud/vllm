# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in eager fallbacks for GLM-5.3-Flash-BF16 on gfx1100.

Keep this module independent of model imports and compiled extensions so its
numerical contracts can also be tested on CPU. Hardware dispatch stays opt-in.
"""

from typing import TYPE_CHECKING

import torch

from vllm import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def enabled(device: torch.device | None = None) -> bool:
    if not envs.VLLM_ROCM_GFX1100_GLM53:
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    if device is None:
        from vllm.platforms.rocm import on_gfx1100

        return on_gfx1100()
    if device.type != "cuda":
        return False
    return (
        torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx1100"
    )


def configure(config: "VllmConfig") -> None:
    """Validate the initial serving envelope before allocating model weights."""
    model = config.model_config
    if model is None:
        return
    text = model.hf_text_config
    if getattr(text, "model_type", None) != "glm5_next_text":
        raise ValueError("VLLM_ROCM_GFX1100_GLM53 requires GLM-5.3-Flash-BF16")
    if model.dtype != torch.bfloat16 or model.quantization is not None:
        raise ValueError("gfx1100 GLM53 requires BF16 weights without quantization")
    if not model.enforce_eager:
        raise ValueError("gfx1100 GLM53 requires --enforce-eager")
    if not 0 < model.max_model_len <= 8192:
        raise ValueError("gfx1100 GLM53 currently requires --max-model-len <= 8192")
    if config.scheduler_config.max_num_seqs != 1:
        raise ValueError("gfx1100 GLM53 currently requires --max-num-seqs 1")
    if config.speculative_config is not None:
        raise ValueError("gfx1100 GLM53 speculative decoding is not validated")
    if config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("gfx1100 GLM53 requires an unquantized BF16 KV cache")
    if getattr(text, "qk_rope_head_dim", None) != 0:
        raise ValueError("gfx1100 GLM53 requires rope-free MLA")
    kernel = config.kernel_config
    if kernel.moe_backend not in ("auto", "triton"):
        raise ValueError("gfx1100 GLM53 requires the Triton MoE backend")
    kernel.moe_backend = "triton"
    kernel.ir_op_priority.rms_norm = ["native"]
    kernel.ir_op_priority.fused_add_rms_norm = ["native"]
    config.compilation_config.cudagraph_mm_encoder = False


def _require_eager(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("gfx1100 Torch fallbacks cannot run during graph capture")


def moe_align(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group original route IDs, preserving EP filtering and padding semantics."""
    _require_eager(topk_ids.device)
    assert block_size > 0 and num_experts > 0
    assert topk_ids.dtype in (torch.int32, torch.int64)
    routes = topk_ids.detach().reshape(-1).cpu().tolist()
    mapping = None
    if expert_map is not None:
        assert expert_map.ndim == 1 and expert_map.numel() == num_experts
        assert expert_map.dtype in (torch.int32, torch.int64)
        mapping = expert_map.cpu().tolist()
        assert all(-1 <= e < num_experts for e in mapping)
    count = len(routes)
    capacity = count + num_experts * (block_size - 1)
    if pad_sorted_ids:
        capacity = (capacity + block_size - 1) // block_size * block_size
    if count < num_experts:
        capacity = min(count * block_size, capacity)
    groups: dict[int, list[int]] = {}
    for route_id, expert in enumerate(routes):
        if expert < 0 or expert >= num_experts:
            continue
        if ignore_invalid_experts and mapping is not None:
            expert = mapping[expert]
            if expert < 0:
                continue
        groups.setdefault(expert, []).append(route_id)
    sorted_ids = [count] * capacity
    expert_ids = [-1] * ((capacity + block_size - 1) // block_size)
    offset = 0
    for expert, ids in sorted(groups.items()):
        blocks = (len(ids) + block_size - 1) // block_size
        assert offset + blocks * block_size <= capacity
        sorted_ids[offset : offset + len(ids)] = ids
        local = (
            mapping[expert]
            if mapping is not None and not ignore_invalid_experts
            else expert
        )
        first = offset // block_size
        expert_ids[first : first + blocks] = [local] * blocks
        offset += blocks * block_size
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=topk_ids.device),
        torch.tensor(expert_ids, dtype=torch.int32, device=topk_ids.device),
        torch.tensor([offset], dtype=torch.int32, device=topk_ids.device),
    )


def moe_sum(
    x: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> None:
    """Accumulate in FP32, excluding uninitialized non-local expert slots."""
    assert x.ndim == 3 and output.shape == (x.shape[0], x.shape[2])
    values = x.float()
    if topk_ids is not None:
        assert topk_ids.shape == x.shape[:2]
        valid = topk_ids >= 0
        if expert_map is not None:
            assert expert_map.ndim == 1 and expert_map.numel() > 0
            valid = valid & (topk_ids < expert_map.numel())
            local = expert_map[topk_ids.clamp(0, expert_map.numel() - 1).long()]
            valid = valid & (local >= 0)
        # Multiplying invalid slots by zero would preserve NaNs.
        values = torch.where(valid.unsqueeze(-1), values, 0.0)
    elif expert_map is not None:
        raise ValueError("expert_map requires topk_ids for pad-aware MoE reduction")
    output.copy_(values.sum(dim=1).to(output.dtype))


def topk_prefill(
    logits: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Write indices relative to each row's start; pad unused entries with -1."""
    _require_eager(logits.device)
    assert starts.ndim == ends.ndim == 1
    assert starts.numel() == ends.numel() == logits.shape[0] == output.shape[0]
    bounds = list(zip(starts.cpu().tolist(), ends.cpu().tolist()))
    assert all(0 <= lo <= hi <= logits.shape[1] for lo, hi in bounds)
    output.fill_(-1)
    for row, (lo, hi) in enumerate(bounds):
        count = min(output.shape[1], hi - lo)
        if count == 0:
            continue
        if hi - lo <= output.shape[1]:
            selected = torch.arange(count, device=logits.device)
        else:
            selected = torch.topk(logits[row, lo:hi], count, sorted=False).indices
        output[row, :count].copy_(selected.to(output.dtype))


def topk_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert next_n > 0 and logits.shape[0] % next_n == 0
    batch = logits.shape[0] // next_n
    lens = seq_lens.to(torch.int64)
    if lens.ndim == 1:
        assert lens.shape[0] == batch
        offsets = torch.arange(next_n, device=lens.device)
        ends = (lens[:, None] - next_n + offsets[None, :] + 1).clamp_min(0)
    else:
        assert tuple(lens.shape) == (batch, next_n)
        ends = lens.clamp_min(0)
    ends = ends.reshape(-1)
    topk_prefill(logits, torch.zeros_like(ends), ends, output)


def concat_and_cache_mla(
    kv: torch.Tensor,
    pe: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    cache_dtype: str,
) -> None:
    """Write unquantized MLA rows, including rope-free and padded inputs."""
    _require_eager(kv.device)
    if cache_dtype not in ("auto", "bfloat16") or cache.dtype != torch.bfloat16:
        raise ValueError("gfx1100 MLA cache fallback requires BF16 cache storage")
    assert kv.ndim == pe.ndim == 2 and cache.ndim == 3 and slots.ndim == 1
    assert kv.dtype == pe.dtype == cache.dtype
    assert slots.numel() <= min(kv.shape[0], pe.shape[0])
    assert cache.shape[-1] == kv.shape[-1] + pe.shape[-1]
    valid = slots >= 0
    selected = slots[valid].long()
    assert (selected < cache.shape[0] * cache.shape[1]).all()
    rows = torch.cat((kv[: slots.numel()], pe[: slots.numel()]), dim=-1)
    cache[selected // cache.shape[1], selected % cache.shape[1]] = rows[valid]
