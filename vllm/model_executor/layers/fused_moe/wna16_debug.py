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


def share_workspace(call: dict, size_bytes: int) -> dict:
    """Place GEMM2 A/C in disjoint views of one allocation, as in modular MoE.

    Reproduce the activation/final-output reservation followed by workspace2.
    This tests allocation size and sharing, not preceding kernels or streams.
    """
    a, c = call["A"], call["C"]
    if a.dtype != c.dtype or a.device != c.device or c.ndim != 3:
        raise ValueError("Expected matching A/C dtype and device and 3D C")

    def span(value):
        if not value.numel():
            raise ValueError("Shared-workspace replay requires nonempty A/C")
        return 1 + sum((n - 1) * s for n, s in zip(value.shape, value.stride()))

    element_size = a.element_size()
    common_bytes = max(span(a), c.shape[0] * c.shape[-1]) * element_size
    c_offset = (common_bytes + 255) // 256 * 256
    required = c_offset + span(c) * element_size
    if size_bytes < required:
        raise ValueError(f"Workspace needs at least {required} bytes, got {size_bytes}")
    storage = torch.empty(size_bytes, dtype=torch.uint8, device=a.device)
    shared = call.copy()
    for name, offset in (("A", 0), ("C", c_offset)):
        source = call[name]
        view = storage[offset : offset + span(source) * element_size].view(a.dtype)
        view = view.as_strided(source.shape, source.stride())
        view.copy_(source)
        shared[name] = view
    return shared


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


def same_bits(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Compare logical tensor bytes, including NaN payloads and signed zeros."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    return torch.equal(
        actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
    )


class ViewPointer:
    """Report view bounds for GEMM2's masked accesses, retaining the allocation.

    Triton HIP accepts ptr_range() instead of using the entire backing storage.
    This is valid only when the kernel's accesses stay inside the tensor view.
    """

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def ptr_range(self) -> int:
        tensor = self.tensor
        if not tensor.numel():
            return 0
        span = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
        return span * tensor.element_size()


def audit_workspace(call: dict, original: dict) -> None:
    """Check restored bytes and the pointers actually received by a GPU kernel."""
    from vllm.triton_utils import tl, triton

    @triton.jit
    def probe(
        A,
        C,
        Copy,
        Addresses,
        expected_a,
        expected_c,
        NA: tl.constexpr,
        NC: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if tl.program_id(0) == 0:
            tl.store(Addresses, A.to(tl.uint64))
            tl.store(Addresses + 1, C.to(tl.uint64))
        # Do not dereference a pointer that the launcher unexpectedly changed.
        if A.to(tl.uint64) == expected_a.to(tl.uint64):
            values = tl.load(A + offsets, offsets < NA, other=0)
            tl.store(Copy + offsets, values, offsets < NA)
        if C.to(tl.uint64) == expected_c.to(tl.uint64):
            tl.store(C + offsets, 3.0, offsets < NC)

    matched = {
        name: same_bits(value, original[name])
        for name, value in call.items()
        if isinstance(value, torch.Tensor)
    }
    print(
        json.dumps({"audit": "restored_bytes", "matches_snapshot": matched}), flush=True
    )
    if not all(matched.values()):
        raise RuntimeError("Restored operands differ from the snapshot before GEMM2")
    a, c = call["A"], call["C"]
    if not a.is_contiguous() or not c.is_contiguous():
        raise ValueError("Pointer probe requires contiguous A/C views")
    copied = torch.full(a.shape, float("nan"), dtype=a.dtype, device=a.device)
    addresses = torch.zeros(2, dtype=torch.uint64, device=a.device)
    c.fill_(float("nan"))
    probe[(triton.cdiv(max(a.numel(), c.numel()), 256),)](
        a,
        c,
        copied,
        addresses,
        a.data_ptr(),
        c.data_ptr(),
        a.numel(),
        c.numel(),
        256,
    )
    torch.accelerator.synchronize()
    observed = addresses.cpu().tolist()
    print(
        json.dumps(
            {
                "audit": "triton_pointer_probe",
                "expected_a": hex(a.data_ptr()),
                "observed_a": hex(observed[0]),
                "expected_c": hex(c.data_ptr()),
                "observed_c": hex(observed[1]),
                "a_read_matches_snapshot": same_bits(copied, original["A"]),
                "a_unchanged": same_bits(a, original["A"]),
                "c_store_3_bad": int((c.cpu() != 3).sum()),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-size-k", type=int, choices=(32, 64, 128))
    parser.add_argument(
        "--view-pointer-range",
        action="store_true",
        help="Report A/C view byte spans to Triton without copying or reallocating",
    )
    parser.add_argument(
        "--audit-workspace",
        action="store_true",
        help="Check restored bytes and Triton pointers/loads/stores before replay",
    )
    parser.add_argument(
        "--workspace-mib",
        type=int,
        default=0,
        help="Share A/C in an allocation of this size; 0 keeps separate allocations",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.workspace_mib < 0 or (args.workspace_mib and args.device != "cuda"):
        parser.error("--workspace-mib must be nonnegative and requires --device cuda")
    if args.audit_workspace and args.device != "cuda":
        parser.error("--audit-workspace requires --device cuda")
    if args.view_pointer_range and args.device != "cuda":
        parser.error("--view-pointer-range requires --device cuda")
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    if snapshot["version"] != 1:
        raise ValueError("Unsupported snapshot version")
    call = snapshot["call"]
    expected = reference(call)
    report = {
        "rank": snapshot["rank"],
        "module": snapshot["module"],
        "workspace_mib": args.workspace_mib,
        "view_pointer_range": args.view_pointer_range,
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
    if args.workspace_mib:
        replay = share_workspace(replay, args.workspace_mib * 1024**2)
        print(
            json.dumps(
                {
                    "workspace_bytes": replay["A"].untyped_storage().nbytes(),
                    "a_offset_bytes": replay["A"].storage_offset()
                    * replay["A"].element_size(),
                    "c_offset_bytes": replay["C"].storage_offset()
                    * replay["C"].element_size(),
                }
            ),
            flush=True,
        )
    if args.block_size_k:
        replay["config"]["BLOCK_SIZE_K"] = args.block_size_k
    if args.audit_workspace:
        audit_workspace(replay, call)
    launch = replay.copy()
    if args.view_pointer_range:
        for name in ("A", "C"):
            launch[name] = ViewPointer(replay[name])
        print(
            json.dumps(
                {
                    "view_pointer_bytes": {
                        name: launch[name].ptr_range() for name in ("A", "C")
                    },
                    "same_addresses": all(
                        launch[name].data_ptr() == replay[name].data_ptr()
                        for name in ("A", "C")
                    ),
                }
            ),
            flush=True,
        )
    compute_type = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[replay["A"].dtype]
    for iteration in range(args.repeats):
        # A NaN sentinel exposes missing writes as well as arithmetic failures.
        replay["C"].fill_(float("nan"))
        if args.audit_workspace:
            print(
                json.dumps(
                    {
                        "audit": "before_gemm2",
                        "replay": iteration,
                        "c_nan_count": int(torch.isnan(replay["C"].cpu()).sum()),
                        "expected_nan_count": replay["C"].numel(),
                        "a_matches_snapshot": same_bits(replay["A"], call["A"]),
                    }
                ),
                flush=True,
            )
        invoke_fused_moe_wna16_triton_kernel(
            **launch,
            compute_type=compute_type,
            use_int8_w8a16=True,
            use_int4_w4a16=False,
        )
        torch.accelerator.synchronize()
        print(json.dumps({"replay": iteration, **compare(replay["C"], expected)}))


if __name__ == "__main__":
    main()
