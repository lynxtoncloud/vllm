# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP collective latency on the serving topology, launched with torchrun."""

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        results = []
        for nbytes in (8192, 65536, 1048576, 16777216):
            for operation in ("all_reduce", "all_gather"):
                x = torch.full(
                    (nbytes // 2,), rank + 1, device="cuda", dtype=torch.bfloat16
                )
                y = torch.empty(size * x.numel(), device="cuda", dtype=x.dtype)

                def call(x=x, y=y, operation=operation):
                    if operation == "all_reduce":
                        dist.all_reduce(x)
                    else:
                        dist.all_gather_into_tensor(y, x)

                call()
                expected = size * (size + 1) // 2
                correct = (
                    (x == expected).all()
                    if operation == "all_reduce"
                    else (
                        y.reshape(size, -1)
                        == torch.arange(1, size + 1, device="cuda")[:, None]
                    ).all()
                )
                if not correct.item():
                    raise RuntimeError(
                        f"{operation} numerical check failed on rank {rank}"
                    )
                x.zero_()
                for _ in range(5):
                    call()
                dist.barrier()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                for _ in range(args.repeats):
                    call()
                end.record()
                end.synchronize()
                ms = torch.tensor(begin.elapsed_time(end) / args.repeats, device="cuda")
                dist.all_reduce(ms, op=dist.ReduceOp.MAX)
                row = {
                    "operation": operation,
                    "input_bytes_per_rank": nbytes,
                    "world_size": size,
                    "slowest_rank_ms": ms.item(),
                }
                results.append(row)
                if rank == 0:
                    print(json.dumps(row), flush=True)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
