# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restart only D with input-stage fences, then replay a saved PD request."""

import argparse
import asyncio
import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import regex as re
from assessment_jobs import write_json
from production_assessment import wait_ready
from run_checks import matches_answer, request, snapshot, transfer_ok

HERE = Path(__file__).resolve().parent


def stop_source_controller(source):
    """Stop only the saved run.py process, using a pidfd to avoid PID reuse."""
    pid_path = source / "pid"
    if not pid_path.exists():
        return
    pid = int(pid_path.read_text().strip())
    if pid <= 1:
        raise ValueError(f"Invalid source controller PID: {pid}")
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            return
        if not any(args):
            return  # Already exited (zombie).
        if os.fsencode(source / "run.py") not in args:
            raise RuntimeError(f"PID {pid} no longer belongs to {source}/run.py")
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        print(f"Stopping source controller PID={pid}", flush=True)
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        if not poll.poll(10000):
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            if not poll.poll(5000):
                raise RuntimeError("Source controller did not exit")
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


def diagnostic_config(cfg):
    cfg = json.loads(json.dumps(cfg))
    if cfg["environment"].get("ENFORCE_EAGER") != "1":
        raise ValueError("Input tracing requires the existing eager configuration")
    cfg["environment"].update(
        AMD_SERIALIZE_KERNEL="0",
        AMD_SERIALIZE_COPY="0",
        HIP_LAUNCH_BLOCKING="0",
        CUDA_LAUNCH_BLOCKING="0",
        VLLM_TRACE_MODEL_INPUTS="1",
        NCCL_DEBUG="INFO",
    )
    return cfg


async def run(args):
    source, output = args.source_dir.resolve(), args.output.resolve()
    if source == output or (output / "config-d.json").exists():
        raise ValueError("Use a new output directory; existing runs are preserved")
    cfg = diagnostic_config(json.loads((source / "config-d.json").read_text()))
    log_root = Path(cfg["log_root"]).resolve()
    if source.parent != log_root or output.parent != log_root:
        raise ValueError("Source and output must be run directories under log_root")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output.name):
        raise ValueError("Output run ID must contain only letters, digits, _ and -")
    addresses = json.loads(
        subprocess.check_output(["ip", "-j", "-4", "addr"], text=True)
    )
    local_ips = {a["local"] for link in addresses for a in link["addr_info"]}
    if cfg["nodes"]["p0"] not in local_ips:
        raise ValueError("Run this controller on P0")
    payload = json.loads((source / "request.json").read_text())
    expected = json.loads((source / "expected.json").read_text())
    if not isinstance(payload.get("prompt"), list) or not payload["prompt"]:
        raise ValueError("Expected a saved token-ID /v1/completions request")
    if set(expected) != {"start", "middle", "end"} or not all(
        isinstance(value, str) for value in expected.values()
    ):
        raise ValueError("Expected three saved needle answers")
    if len(payload["prompt"]) + payload["max_tokens"] > int(
        cfg["environment"]["MAX_MODEL_LEN"]
    ):
        raise ValueError("Saved request exceeds configured context")
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    output.mkdir(parents=True, exist_ok=True)

    def state(stage, **details):
        write_json(
            output / "status.json", dict(stage=stage, time=time.time(), **details)
        )
        print(stage, details, flush=True)

    config_path = output / "config-d.json"

    def control(action, role, run_id, config, check=True):
        return subprocess.run(
            [
                sys.executable,
                str(HERE / "assessment_jobs.py"),
                action,
                "engines",
                "--only-role",
                role,
                "--run-id",
                run_id,
                "--config",
                str(config),
            ],
            check=check,
        )

    try:
        state("检查P与代理，保存同一条输入")
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            for port, route in ((8001, "health"), (8000, "healthcheck")):
                response = await client.get(
                    f"http://{cfg['nodes']['p0']}:{port}/{route}"
                )
                response.raise_for_status()
        write_json(config_path, cfg)
        write_json(output / "request.json", payload)
        write_json(output / "expected.json", expected)
        (output / "client-commit.txt").write_text(
            subprocess.check_output(
                ["git", "-C", cfg["repo"], "rev-parse", "HEAD"], text=True
            )
        )
        state("停止旧同步诊断控制进程与D组")
        stop_source_controller(source)
        # Stop both hosts even if the first reports delayed VRAM release.
        for role in ("d0", "d1"):
            control("stop", role, source.name, source / "config-d.json", check=False)
        for role in ("d0", "d1"):
            control("gpu-clean", role, source.name, source / "config-d.json")
        state("启动D组，保留P与代理")
        for role in ("d0", "d1"):
            control("start", role, output.name, config_path)
        state("等待D组就绪", timeout_s=3600)
        await wait_ready(cfg, 3600)
        state("重放原长输入", input_tokens=len(payload["prompt"]))
        async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
            decoder = f"http://{cfg['nodes']['d0']}:8002"
            before = await snapshot(client, decoder, output / "D-before.prom")
            row = await request(
                client,
                f"http://{cfg['nodes']['p0']}:8000",
                "/v1/completions",
                payload,
                14400,
            )
            write_json(output / "response.json", row)
            try:
                after = await snapshot(client, decoder, output / "D-after.prom")
                kv_ok = transfer_ok(before, after)
            except httpx.HTTPError as error:
                kv_ok = False
                row["metrics_error"] = str(error)
            passed = bool(
                row["success"]
                and matches_answer(row["text"], expected)
                and row.get("input_tokens") == len(payload["prompt"])
                and kv_ok
            )
            row.update(transfer_ok=kv_ok, expected=expected, passed=passed)
            write_json(output / "response.json", row)
        state("复现结束", passed=passed)
        return 0 if passed else 1
    except Exception as error:
        state("流程异常停止", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # One replay controller across different output directories on P0.
    with (args.source_dir.parent / "input-trace.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(asyncio.run(run(args)))
