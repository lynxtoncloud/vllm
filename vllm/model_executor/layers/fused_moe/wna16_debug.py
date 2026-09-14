# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture failing W8A16 GEMM2 operands and replay without loading a model."""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def save_failure(directory: str, *, rank: int, module: str, **call) -> Path | None:
    if torch.isfinite(call["C"]).all().item():
        return None
    tensors = {k: v for k, v in call.items() if isinstance(v, torch.Tensor)}
    snapshot = {
        "version": 1,
        "rank": rank,
        "module": module,
        "device": str(call["C"].device),
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "strides": {k: tuple(v.stride()) for k, v in tensors.items()},
        "call": {
            k: v.detach().to("cpu").contiguous().clone() if k in tensors else v
            for k, v in call.items()
        },
    }
    path = Path(directory) / f"gemm2-rank{rank}-pid{os.getpid()}-{time.time_ns()}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(snapshot, temporary)
    temporary.replace(path)
    print(f"GLM GEMM2 capture: {path}", flush=True)
    return path


def reference(call: dict) -> torch.Tensor:
    """CPU float64 dot products after matching the kernel's BF16 weight cast.

    Interpret actual alignment metadata, including zero output for remote EP
    experts. Reject duplicate/missing assignments instead of hiding bad metadata.
    """
    a, b, scales = call["A"], call["B"], call["B_scale"]
    output = torch.zeros_like(call["C"], dtype=torch.float64)
    flat = output.view(-1, output.shape[-1])
    seen = torch.zeros(flat.shape[0], dtype=torch.bool)
    block_m = call["config"]["BLOCK_SIZE_M"]
    padded = int(call["num_tokens_post_padded"].item())
    ids, experts = call["sorted_token_ids"], call["expert_ids"]
    if padded < 0 or padded > ids.numel() or padded % block_m:
        raise ValueError("Invalid padded token count")
    for offset in range(0, padded, block_m):
        expert = int(experts[offset // block_m])
        rows = ids[offset : offset + block_m].long()
        if (rows < 0).any():
            raise ValueError("Negative sorted token id")
        rows = rows[rows < flat.shape[0]]
        if rows.unique().numel() != rows.numel() or seen[rows].any():
            raise ValueError("Duplicate token assignment")
        seen[rows] = True
        if expert == -1 or not rows.numel():
            continue
        if not 0 <= expert < b.shape[0]:
            raise ValueError(f"Invalid local expert id {expert}")
        group = call["block_shape"][1]
        scale = scales[expert].float().repeat_interleave(group, dim=-1)
        zp = call["B_zp"]
        zero = 128 if zp is None else zp[expert].float().repeat_interleave(group, -1)
        weight = ((b[expert].float() - zero) * scale).to(a.dtype).double()
        values = a[rows // call["top_k"]].double() @ weight.T
        if call["mul_routed_weight"]:
            values *= call["topk_weights"].reshape(-1)[rows].double()[:, None]
        flat[rows] = values
    if not seen.all():
        raise ValueError(f"Missing {int((~seen).sum())} token assignments")
    return output


def restore_call(snapshot: dict, device: str) -> dict:
    call = {}
    for name, value in snapshot["call"].items():
        if isinstance(value, torch.Tensor):
            restored = torch.empty_strided(
                value.shape, snapshot["strides"][name], dtype=value.dtype, device=device
            )
            restored.zero_()
            restored.copy_(value)
            call[name] = restored
        else:
            call[name] = value.copy() if isinstance(value, dict) else value
    return call


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    rounded_reference = expected.to(actual.dtype)
    actual = actual.cpu().double()
    valid = torch.isfinite(actual) & torch.isfinite(expected)
    return {
        "nonfinite": int((~torch.isfinite(actual)).sum()),
        "reference_nonfinite": int((~torch.isfinite(expected)).sum()),
        "reference_nonfinite_after_output_cast": int(
            (~torch.isfinite(rounded_reference)).sum()
        ),
        "max_abs_error_finite": (
            float((actual[valid] - expected[valid]).abs().max())
            if valid.any()
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-size-k", type=int, choices=(32, 64, 128))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    if snapshot["version"] != 1:
        raise ValueError("Unsupported snapshot version")
    call = snapshot["call"]
    expected = reference(call)
    report = {
        "rank": snapshot["rank"],
        "module": snapshot["module"],
        "config": call["config"],
        "strides": snapshot["strides"],
        "original": compare(call["C"], expected),
        "operands": {
            name: {
                "shape": list(value.shape),
                "nonfinite": int((~torch.isfinite(value)).sum()),
                "max_abs": (
                    float(value.double().abs().max())
                    if value.numel() and torch.isfinite(value).all()
                    else None
                ),
            }
            for name, value in call.items()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        },
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.device == "cpu":
        return
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_wna16_triton_kernel,
    )
    from vllm.triton_utils import tl

    replay = restore_call(snapshot, "cuda")
    if args.block_size_k:
        replay["config"]["BLOCK_SIZE_K"] = args.block_size_k
    compute_type = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[replay["A"].dtype]
    for iteration in range(args.repeats):
        # A NaN sentinel exposes missing writes as well as arithmetic failures.
        replay["C"].fill_(float("nan"))
        invoke_fused_moe_wna16_triton_kernel(
            **replay,
            compute_type=compute_type,
            use_int8_w8a16=True,
            use_int4_w4a16=False,
        )
        torch.accelerator.synchronize()
        print(json.dumps({"replay": iteration, **compare(replay["C"], expected)}))


if __name__ == "__main__":
    main()
