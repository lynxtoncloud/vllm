# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline W8A16 tuning. Only numerically checked exact shapes are exported."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from vllm.model_executor.kernels.linear.mixed_precision.triton_w8a16 import (
    _w8a16_gemm,
)
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shape", nargs=2, type=int, action="append", required=True, metavar=("N", "K")
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 512, 2048],
    )
    args = parser.parse_args()
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("Run on a ROCm GPU with no serving workload")
    if any(n < 1 or k < 128 or k % 128 for n, k in args.shape):
        parser.error("N must be positive; K must be a positive multiple of 128")
    if any(m < 1 for m in args.batch_sizes):
        parser.error("Batch sizes must be positive")
    torch.manual_seed(123)
    properties = torch.cuda.get_device_properties(0)
    result = {
        "version": 1,
        "device": properties.name,
        "arch": properties.gcnArchName,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "configs": {},
        "measurements": {},
    }
    baseline = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=32, num_warps=4)
    candidates = [baseline] + [
        dict(BLOCK_M=m, BLOCK_N=n, BLOCK_K=k, num_warps=w)
        for m, n, k, w in itertools.product(
            (16, 32, 64), (32, 64, 128), (32, 64, 128), (4, 8)
        )
        if (m, n, k, w) != (16, 32, 32, 4)
    ]
    for n, k in args.shape:
        quant = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int32)
        packed = torch.zeros(n, k // 4, device="cuda", dtype=torch.int32)
        for shift in range(4):
            packed |= (quant[:, shift::4] + 128) << (shift * 8)
        scales = (torch.rand(n, k // 128, device="cuda") * 0.001).bfloat16()
        weight = (
            quant.float() * scales.float().repeat_interleave(128, dim=1)
        ).bfloat16()
        for m in args.batch_sizes:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            expected = x.float() @ weight.float().T
            output = torch.empty(m, n, device="cuda", dtype=x.dtype)
            measurements = []
            for config in candidates:

                def call(
                    x=x,
                    packed=packed,
                    scales=scales,
                    output=output,
                    m=m,
                    n=n,
                    k=k,
                    config=config,
                ):
                    _w8a16_gemm[
                        (
                            triton.cdiv(m, config["BLOCK_M"]),
                            triton.cdiv(n, config["BLOCK_N"]),
                        )
                    ](
                        x,
                        packed,
                        scales,
                        output,
                        m,
                        n,
                        k,
                        *x.stride(),
                        GROUP_SIZE=128,
                        **config,
                    )

                try:
                    call()
                    torch.testing.assert_close(
                        output.float(), expected, atol=0.05, rtol=0.02
                    )
                    milliseconds = triton.testing.do_bench(call)
                    measurements.append({"config": config, "ms": milliseconds})
                except (AssertionError, triton.runtime.errors.OutOfResources) as error:
                    measurements.append({"config": config, "error": str(error)})
            if "error" in measurements[0]:
                raise RuntimeError(
                    f"Baseline failed numerical check: {measurements[0]}"
                )
            valid = [entry for entry in measurements if "ms" in entry]
            best = min(valid, key=lambda entry: entry["ms"])
            if best["ms"] >= measurements[0]["ms"] * 0.95:
                best = measurements[0]
            key = f"{x.dtype}:{m}:{n}:{k}"
            result["configs"][key] = best["config"]
            result["measurements"][key] = measurements
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(args.output)
            print(
                json.dumps(
                    {"shape": key, "best": best, "baseline_ms": measurements[0]["ms"]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
