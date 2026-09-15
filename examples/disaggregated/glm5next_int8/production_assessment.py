# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequential performance A/B runs supervised by assessment_jobs on P0.

Every profile gets fresh engines and its own immutable run directory. A failed
functional check stops the campaign. Multimodal diagnosis is deliberately kept
out of this text performance campaign, and remains a separate acceptance gate.
"""

import argparse
import asyncio
import copy
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from assessment_suite import health, run, save

HERE = Path(__file__).resolve().parent
PROFILES = [
    ("baseline", {"ENFORCE_EAGER": "1", "VLLM_ROCM_USE_TRITON_MQA_LOGITS": "0"}),
    ("indexer", {"ENFORCE_EAGER": "1", "VLLM_ROCM_USE_TRITON_MQA_LOGITS": "1"}),
    (
        "graph64",
        {
            "ENFORCE_EAGER": "0",
            "VLLM_ROCM_USE_TRITON_MQA_LOGITS": "1",
            "GRAPH_CAPTURE_SIZES": "1,2,4,8,16,32,64",
        },
    ),
    (
        "prefill2048",
        {
            "ENFORCE_EAGER": "0",
            "VLLM_ROCM_USE_TRITON_MQA_LOGITS": "1",
            "GRAPH_CAPTURE_SIZES": "1,2,4,8,16,32,64",
            "MAX_BATCHED_TOKENS": "2048",
        },
    ),
    (
        "prefill8192",
        {
            "ENFORCE_EAGER": "0",
            "VLLM_ROCM_USE_TRITON_MQA_LOGITS": "1",
            "GRAPH_CAPTURE_SIZES": "1,2,4,8,16,32,64",
            "MAX_BATCHED_TOKENS": "8192",
        },
    ),
]


def profiles(cfg):
    selected = cfg.get("production_profiles", [name for name, _ in PROFILES])
    known = dict(PROFILES)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in known for name in selected)
    ):
        raise ValueError(
            f"production_profiles must select unique names from {list(known)}"
        )
    for name in selected:
        candidate = copy.deepcopy(cfg)
        candidate["fail_fast_functional"] = True
        candidate["cases"] = cfg.get("production_cases", ["text2k", "text32k"])
        if not candidate["cases"] or any(
            case not in ("text2k", "text32k", "text128k", "text512k", "text1m")
            for case in candidate["cases"]
        ):
            raise ValueError(
                "Production performance campaign currently selects text cases"
            )
        # Each candidate is relative to the same baseline; inherited debug flags
        # must not change synchronization or force graph rejection.
        candidate["environment"].update(
            {
                "MAX_BATCHED_TOKENS": "512",
                "ENFORCE_EAGER": "1",
                "GRAPH_CAPTURE_SIZES": "1,2,4,8,16,32,64",
                "VLLM_GLM5NEXT_CHECK_FINITE": "0",
                "VLLM_GLM5NEXT_TRACE_VISION": "0",
                "VLLM_GLM5NEXT_ISOLATE_GEMM2": "0",
            }
        )
        candidate["environment"].pop("VLLM_GLM5NEXT_DUMP_DIR", None)
        candidate["environment"].update(known[name])
        yield name, candidate


async def wait_ready(cfg, timeout):
    urls = [
        f"http://{cfg['nodes'][role]}:{port}"
        for role, port in (("p0", 8000), ("p0", 8001), ("d0", 8002))
    ]
    deadline = time.monotonic() + timeout
    last = None
    async with httpx.AsyncClient(trust_env=False) as client:
        while time.monotonic() < deadline:
            try:
                await health(client, urls)
                return
            except (httpx.HTTPError, ValueError) as error:
                last = error
            await asyncio.sleep(10)
    raise RuntimeError(f"Engine/proxy readiness timeout: {last}")


def control(cfg_path, run_id, action, kind, check=True):
    return subprocess.run(
        [
            sys.executable,
            str(HERE / "assessment_jobs.py"),
            action,
            kind,
            "--run-id",
            run_id,
            "--config",
            str(cfg_path),
        ],
        check=check,
    )


def qualify_kernels(cfg, out):
    """Do not let skipped GPU tests qualify an optimized serving profile."""
    tests = [
        ("indexer", "tests/v1/attention", "test_rocm_glm5next_sparse.py", "batched_"),
        ("linear", "tests/kernels/quantization", "test_triton_w8a16.py", ""),
    ]
    env = os.environ.copy()
    env.update(HIP_VISIBLE_DEVICES="0", VLLM_ROCM_W8A16_CONFIG="")
    for name, folder, file, expression in tests:
        xml_path = out / f"{name}-qualification.xml"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--confcutdir={folder}",
            f"{folder}/{file}",
            f"--junitxml={xml_path}",
        ]
        if expression:
            cmd += ["-k", expression]
        log_path = out / f"{name}-qualification.log"
        try:
            with log_path.open("w") as log:
                subprocess.run(
                    cmd,
                    cwd=cfg["repo"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except subprocess.CalledProcessError as exc:
            tail = log_path.read_text(errors="replace")[-12000:]
            raise RuntimeError(
                f"GPU qualification failed: {name} (exit {exc.returncode}). "
                f"Full log: {log_path}\n{tail}"
            ) from exc
        cases = list(ET.parse(xml_path).iter("testcase"))
        if not cases or any(
            len(case)
            and any(child.tag in ("skipped", "error", "failure") for child in case)
            for case in cases
        ):
            raise RuntimeError(f"GPU qualification incomplete: {xml_path}")


async def campaign(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    base_id = cfg["production_run_id"]
    manifest = {
        "config": cfg,
        "commit": subprocess.check_output(
            ["git", "-C", cfg["repo"], "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Campaign code/config changed; use a new run ID")
    save(manifest_path, manifest)
    gate_config = out / "gate-config.json"
    save(gate_config, cfg)
    control(gate_config, base_id, "gpu-check", "engines")
    qualify_kernels(cfg, out)
    results = []
    for name, candidate in profiles(cfg):
        run_id = f"{base_id}-{name}"
        directory = out / name
        directory.mkdir(exist_ok=True)
        path = directory / "config.json"
        save(path, candidate)
        save(out / "active.json", {"run_id": run_id, "config": str(path)})
        data = Path(cfg["log_root"]) / run_id / "p0/test/data"
        save(
            out / "progress.json",
            {
                "status": "running",
                "profile": name,
                "run_id": run_id,
                "time": time.time(),
            },
        )
        try:
            # An active prior run is rejected by the existing ownership checks.
            control(path, run_id, "start", "monitors")
            control(path, run_id, "start", "engines")
            control(path, run_id, "start", "proxy")
            await wait_ready(candidate, cfg.get("production_start_timeout", 3600))
            await run(candidate, data)
            report = json.loads((data / "progress.json").read_text())
            if report.get("status") != "complete":
                raise RuntimeError(f"Profile failed acceptance: {data}")
            results.append(
                {"profile": name, "run_id": run_id, "data": str(data), "passed": True}
            )
        finally:
            # Preserve monitors until engines have exited, including stop/interrupt.
            cleanup_errors = []
            for kind in ("proxy", "engines", "monitors"):
                if control(path, run_id, "stop", kind, check=False).returncode:
                    cleanup_errors.append(kind)
            if control(path, run_id, "collect", "all", check=False).returncode:
                cleanup_errors.append("collect")
            if cleanup_errors:
                message = (
                    f"Cleanup/collection incomplete for {run_id}: {cleanup_errors}"
                )
                if sys.exc_info()[0] is None:
                    raise RuntimeError(message)
                print(message, file=sys.stderr, flush=True)
        save(out / "results.json", results)
    save(
        out / "progress.json",
        {
            "status": "completed",
            "time": time.time(),
            "profiles": results,
            "scope": "text performance; multimodal gate pending",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())

    def stop(signum, frame):
        raise KeyboardInterrupt("Campaign stopped by operator")

    signal.signal(signal.SIGTERM, stop)
    try:
        asyncio.run(campaign(cfg, args.output))
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        save(
            args.output / "progress.json",
            {
                "status": "stopped_with_error",
                "error": str(error),
                "time": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
