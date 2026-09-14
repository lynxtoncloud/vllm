# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe local UCX registration without loading the model or contacting peers."""

import argparse
import importlib.metadata
import json
import os
import resource
import socket
import subprocess
import sys
import uuid
from pathlib import Path


def register_buffer(agent, tensor, memory_type):
    device = tensor.get_device() if memory_type == "VRAM" else 0
    storage = tensor.untyped_storage()
    descs = agent.get_reg_descs(
        [(storage.data_ptr(), storage.nbytes(), device, "")], memory_type
    )
    agent.register_memory(descs, backends=["UCX"])
    agent.deregister_memory(descs, backends=["UCX"])


def probe(device, size_mib, memory_type):
    # Match the connector's environment before importing NIXL.
    os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")
    import torch
    from nixl_rocm._api import nixl_agent, nixl_agent_config

    if not torch.version.hip:
        raise RuntimeError("ROCm PyTorch required")
    torch.accelerator.set_device_index(device)
    tensor = torch.empty(
        size_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=f"cuda:{device}" if memory_type == "VRAM" else "cpu",
        pin_memory=memory_type == "DRAM",
    )
    tensor.zero_()
    torch.accelerator.synchronize()
    agent = nixl_agent(
        f"pd-preflight-{uuid.uuid4().hex}",
        nixl_agent_config(num_threads=4, capture_telemetry=True),
    )
    maps = Path("/proc/self/maps")
    if maps.exists():
        libs = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if any(lib in line for lib in ("libucp", "libuct", "libucs"))
            }
        )
        print("Loaded UCX libraries:", json.dumps(libs), flush=True)
    register_buffer(agent, tensor, memory_type)
    # Destroy the agent before its backing tensor goes out of scope.
    del agent
    print(f"PASS device={device} memory={memory_type} MiB={size_mib}", flush=True)


def run_matrix(args):
    failures = []
    for memory_type in args.memory:
        devices = args.devices if memory_type == "VRAM" else args.devices[:1]
        for device in devices:
            for size in args.sizes_mib:
                label = f"device={device} memory={memory_type} MiB={size}"
                print(f"START {label}", flush=True)
                # A fresh process isolates partial native registration failures
                # and allocator state; UCX stderr remains visible in the log.
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--devices",
                    str(device),
                    "--sizes-mib",
                    str(size),
                    "--memory",
                    memory_type,
                ]
                try:
                    result = subprocess.run(cmd, timeout=args.timeout)
                    failed = result.returncode != 0
                except subprocess.TimeoutExpired:
                    failed = True
                    print(f"TIMEOUT {label}", flush=True)
                if failed:
                    failures.append(label)
                    print(f"FAIL {label}", flush=True)
    print(json.dumps({"passed": not failures, "failures": failures}), flush=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[8, 2048])
    parser.add_argument(
        "--memory", choices=("DRAM", "VRAM"), nargs="+", default=["DRAM", "VRAM"]
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.devices) < 0 or min(args.sizes_mib) < 1 or args.timeout <= 0:
        parser.error("Devices must be nonnegative; sizes and timeout must be positive")
    if args.worker:
        probe(args.devices[0], args.sizes_mib[0], args.memory[0])
        return 0
    versions = {}
    for name in ("torch", "nixl-rocm", "vllm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not in package metadata"
    print(
        json.dumps(
            {
                "host": socket.gethostname(),
                "kernel": os.uname().release,
                "versions": versions,
                "memlock_bytes": resource.getrlimit(resource.RLIMIT_MEMLOCK),
                "environment": {
                    key: value
                    for key, value in sorted(os.environ.items())
                    if key.startswith(("UCX_", "NIXL_", "HIP_VISIBLE_DEVICES"))
                    or (key.startswith("PYTORCH_") and "ALLOC_CONF" in key)
                },
            }
        ),
        flush=True,
    )
    return run_matrix(args)


if __name__ == "__main__":
    sys.exit(main())
