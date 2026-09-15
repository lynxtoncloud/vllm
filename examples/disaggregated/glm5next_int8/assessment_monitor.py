# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only Linux host/GPU/network sampler; never initializes a GPU context."""

import argparse
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def number(path):
    try:
        return int(read(path))
    except (TypeError, ValueError):
        return None


def capture(command, timeout=8):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        return {
            "code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def gpu_samples():
    result = {}
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name or read(card / "device/vendor") != "0x1002":
            continue
        device = card / "device"
        gpu = {"pci": device.resolve().name}
        gpu["clocks_and_link"] = {
            name: read(device / name)
            for name in (
                "pp_dpm_sclk",
                "pp_dpm_mclk",
                "current_link_speed",
                "current_link_width",
            )
        }
        for field in (
            "gpu_busy_percent",
            "mem_busy_percent",
            "mem_info_vram_used",
            "mem_info_vram_total",
        ):
            gpu[field] = number(device / field)
        for hwmon in (device / "hwmon").glob("hwmon*"):
            gpu["sensors"] = {
                p.name: number(p)
                for pattern in (
                    "power*_average",
                    "power*_input",
                    "power*_cap",
                    "temp*_input",
                    "fan*_input",
                )
                for p in hwmon.glob(pattern)
            }
        result[card.name] = gpu
    return result


def network_samples():
    return {
        nic.name: {
            "speed_mbps": number(nic / "speed"),
            "operstate": read(nic / "operstate"),
            "counters": {p.name: number(p) for p in (nic / "statistics").glob("*")},
        }
        for nic in Path("/sys/class/net").iterdir()
    }


def rdma_samples():
    return {
        str(port): {
            "rate": read(port / "rate"),
            "counters": {
                p.name: number(p)
                for folder in ("counters", "hw_counters")
                for p in (port / folder).glob("*")
            },
        }
        for port in Path("/sys/class/infiniband").glob("*/ports/*")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    interval = float(cfg["monitor_interval"])
    if interval < 1:
        raise ValueError("monitor_interval must be >= 1 second")
    commands = [
        ["uname", "-a"],
        ["lscpu"],
        ["free", "-b"],
        ["df", "-h"],
        ["ip", "-s", "link"],
        ["ip", "addr"],
        ["ip", "route"],
        ["timedatectl"],
        ["ethtool", cfg["interface"]],
        ["ethtool", "-i", cfg["interface"]],
        ["rdma", "link"],
        ["ibv_devinfo"],
        ["git", "-C", cfg["repo"], "rev-parse", "HEAD"],
        ["git", "-C", cfg["repo"], "diff", "--stat"],
        [
            "sha256sum",
            *(
                str(Path(cfg["model"]) / name)
                for name in (
                    "config.json",
                    "tokenizer_config.json",
                    "model.safetensors.index.json",
                )
            ),
        ],
        [
            str(Path(cfg["repo"]) / ".venv/bin/python"),
            "-c",
            (
                "from importlib.metadata import distributions; "
                "print([(d.metadata['Name'], d.version) for d in distributions() "
                "if d.metadata['Name'].lower().replace('_','-') in "
                "('torch','triton','pytorch-triton-rocm','nixl-rocm','vllm')])"
            ),
        ],
        ["/data/opt/ucx/bin/ucx_info", "-v"],
        ["rocm-smi", "--showproductname", "--showdriverversion"],
    ]
    inventory = {
        "hostname": socket.gethostname(),
        "time": time.time(),
        "commands": {" ".join(cmd): capture(cmd) for cmd in commands},
    }
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    last_slow = 0
    with (args.output / "samples.jsonl").open("a", buffering=1) as out:
        while True:
            started = time.monotonic()
            sample = {
                "time": time.time(),
                "monotonic": started,
                "hostname": socket.gethostname(),
                "gpu": gpu_samples(),
                "network": network_samples(),
                "rdma": rdma_samples(),
                "proc": {
                    name: read("/proc/" + name)
                    for name in (
                        "stat",
                        "meminfo",
                        "loadavg",
                        "vmstat",
                        "diskstats",
                        "pressure/cpu",
                        "pressure/io",
                        "pressure/memory",
                    )
                },
            }
            if started - last_slow >= 60:
                sample["processes"] = capture(
                    ["ps", "-eo", "pid,ppid,stat,pcpu,pmem,rss,comm"]
                )
                sample["sockets"] = capture(["ss", "-s"])
                sample["kernel"] = capture(["dmesg", "--since", "1 minute ago"])
                # Local BMC only; no credentials or remote BMC access required.
                if shutil.which("ipmitool"):
                    sample["host_power"] = capture(
                        ["ipmitool", "dcmi", "power", "reading"]
                    )
                last_slow = started
            out.write(json.dumps(sample) + "\n")
            time.sleep(max(0, interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
