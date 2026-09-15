# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detached jobs with explicit start/stop/status; cluster commands run on P0."""

import argparse
import fcntl
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import regex as re

HERE = Path(__file__).resolve().parent
KINDS = {
    "production": ("production", ("p0",)),
    "communication": ("collective", ("p0", "p1", "d0", "d1")),
    "engines": ("engine", ("p0", "p1", "d0", "d1")),
    "monitors": ("monitor", ("p0", "p1", "d0", "d1")),
    "proxy": ("proxy", ("p0",)),
    "test": ("test", ("p0",)),
}


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "state": fields[0],
            "start_ticks": fields[19],
            "session": int(fields[3]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (OSError, ValueError, IndexError):
        return None


def owned(state):
    actual = identity(state.get("pid", -1))
    return bool(
        actual
        and actual["state"] != "Z"
        and actual["start_ticks"] == state.get("start_ticks")
        and actual["boot_id"] == state.get("boot_id")
        and actual["session"] == state["pid"]
    )


def command(cfg, role, kind, directory):
    repo = Path(cfg["repo"])
    python = str(repo / ".venv/bin/python")
    script = repo / "examples/disaggregated/glm5next_int8"
    env = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "VLLM_ROCM_GFX1100_GLM53",
        "VLLM_GLM5NEXT_CHECK_FINITE",
        "VLLM_GLM5NEXT_TRACE_VISION",
        "VLLM_GLM5NEXT_DUMP_DIR",
        "VLLM_GLM5NEXT_ISOLATE_GEMM2",
        "VLLM_ROCM_USE_TRITON_MQA_LOGITS",
        "VLLM_ROCM_W8A16_CONFIG",
        "GRAPH_CAPTURE_SIZES",
        "MAX_NUM_SEQS",
    ):
        env.pop(name, None)
    env.update(cfg["environment"])
    env.update(
        PYTHONUNBUFFERED="1",
        MODEL_DIR=cfg["model"],
        LOG_DIR=str(directory / "engine-logs"),
    )
    env.update({f"{node.upper()}_IP": ip for node, ip in cfg["nodes"].items()})
    if kind == "engine":
        return ["bash", str(script / "launch_pd.sh"), role], env
    if kind == "collective":
        leader = "p0" if role.startswith("p") else "d0"
        env.update(
            HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
            NCCL_SOCKET_IFNAME=cfg["interface"],
            GLOO_SOCKET_IFNAME=cfg["interface"],
        )
        return [
            python,
            "-m",
            "torch.distributed.run",
            "--nnodes=2",
            "--nproc-per-node=8",
            f"--node-rank={int(role.endswith('1'))}",
            f"--master-addr={cfg['nodes'][leader]}",
            f"--master-port={29601 if leader == 'p0' else 29602}",
            str(repo / "benchmarks/kernels/benchmark_rocm_collectives.py"),
            "--output",
            str(directory / "data/collectives.json"),
        ], env
    if kind == "proxy":
        return [
            python,
            "tests/v1/kv_connector/nixl_integration/toy_proxy_server.py",
            "--host",
            cfg["nodes"]["p0"],
            "--port",
            "8000",
            "--prefiller-hosts",
            cfg["nodes"]["p0"],
            "--prefiller-ports",
            "8001",
            "--decoder-hosts",
            cfg["nodes"]["d0"],
            "--decoder-ports",
            "8002",
        ], env
    name = {
        "monitor": "assessment_monitor.py",
        "test": "assessment_suite.py",
        "production": "production_assessment.py",
    }[kind]
    return [
        python,
        str(script / name),
        "--config",
        str(directory / "config.json"),
        "--output",
        str(directory / "data"),
    ], env


def worker(cfg, args, directory):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    stopping = False

    def terminate(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, terminate)
    cmd, env = command(cfg, args.role, args.kind, directory)
    names = set(cfg["environment"]) | {
        "MAX_NUM_SEQS",
        "UCX_TLS",
        "UCX_NET_DEVICES",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "HSA_ENABLE_SDMA",
        "VLLM_GLM5NEXT_DUMP_DIR",
        "VLLM_GLM5NEXT_ISOLATE_GEMM2",
    }
    write_json(
        directory / "launch-env.json", {name: env.get(name) for name in sorted(names)}
    )
    started = time.time()
    state = {
        "pid": os.getpid(),
        **identity(os.getpid()),
        "started": started,
        "hostname": socket.gethostname(),
        "command": cmd,
    }
    write_json(directory / "state.json", state)
    child = subprocess.Popen(cmd, cwd=cfg["repo"], env=env)
    while child.poll() is None and not stopping:
        time.sleep(1)
    if stopping:
        # The supervisor ignores TERM itself; all children inherit its session.
        os.killpg(os.getpid(), signal.SIGTERM)
        deadline = time.monotonic() + 20
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
    state.update(finished=time.time(), exit_code=child.poll(), stopped=stopping)
    write_json(directory / "state.json", state)
    # Clean up only this new session, including orphaned distributed workers.
    os.killpg(os.getpid(), signal.SIGKILL)


def local(cfg, args):
    root = Path(cfg["log_root"]) / args.run_id / args.role
    directory = root / args.kind
    directory.mkdir(parents=True, exist_ok=True)
    if args.action in ("gpu-check", "gpu-clean"):
        from assessment_gpu import run

        if not run(cfg, args.role, args.action, args.pids, root / "gpu-checks"):
            raise SystemExit(
                "VRAM insufficient, ownership unresolved or cleanup incomplete"
            )
        return
    if args.action == "worker":
        worker(cfg, args, directory)
        return
    # One active job of each kind per host, even across different run IDs.
    lock_dir = Path(cfg["log_root"]) / ".jobs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{args.role}-{args.kind}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if args.action == "status":
            print(
                json.dumps(
                    {
                        "role": args.role,
                        "kind": args.kind,
                        "running": owned(state),
                        **state,
                    }
                )
            )
            progress = directory / "data/progress.json"
            if progress.exists():
                print(progress.read_text())
            return
        if args.action == "stop":
            if owned(state):
                os.killpg(state["pid"], signal.SIGTERM)
                deadline = time.monotonic() + 30
                while owned(state) and time.monotonic() < deadline:
                    time.sleep(0.2)
                if owned(state):
                    os.killpg(state["pid"], signal.SIGKILL)
            print(f"Stopped/absent: {args.role}/{args.kind} {args.run_id}")
            if args.kind == "engine":
                from assessment_gpu import run

                if not run(cfg, args.role, "gpu-check", [], root / "gpu-checks"):
                    raise SystemExit("Managed job stopped; VRAM still insufficient")
            return
        for old in Path(cfg["log_root"]).glob(f"*/{args.role}/{args.kind}/state.json"):
            if owned(json.loads(old.read_text())):
                raise RuntimeError(f"Job already running: {old}")
        config_path = directory / "config.json"
        current_commit = subprocess.check_output(
            ["git", "-C", cfg["repo"], "rev-parse", "HEAD"], text=True
        ).strip()
        if current_commit != cfg.get("expected_commit", current_commit):
            raise RuntimeError(f"Code differs from P0: {current_commit}")
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            raise ValueError("Configuration changed; use a new run ID")
        if args.kind in ("engine", "collective"):
            from assessment_gpu import run

            if not run(cfg, args.role, "gpu-check", [], root / "gpu-checks"):
                raise RuntimeError("VRAM check failed; inspect owners before starting")
        write_json(config_path, cfg)
        cmd = [
            sys.executable,
            str(HERE / "assessment_jobs.py"),
            "worker",
            args.kind,
            "--local",
            "--role",
            args.role,
            "--run-id",
            args.run_id,
            "--config",
            str(config_path),
        ]
        with (directory / "console.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
            )
        for _ in range(50):
            if state_path.exists():
                new = json.loads(state_path.read_text())
                if new.get("pid") == process.pid:
                    print(
                        f"Started {args.role}/{args.kind} PID={process.pid} {directory}"
                    )
                    return
            if process.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError(f"Worker did not start; see {directory}/console.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "start",
            "stop",
            "status",
            "collect",
            "worker",
            "gpu-check",
            "gpu-clean",
        ),
    )
    parser.add_argument(
        "kind", choices=(*KINDS, "all", "engine", "monitor", "collective")
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=HERE / "assessment_config.json")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--role", choices=("p0", "p1", "d0", "d1"))
    parser.add_argument("--only-role", choices=("p0", "p1", "d0", "d1"))
    parser.add_argument("--pids", type=int, nargs="+", default=[])
    args = parser.parse_args()
    if any(pid <= 1 for pid in args.pids):
        parser.error("Worker PIDs must be > 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("run-id must contain only letters, digits, _ and -")
    cfg = json.loads(args.config.read_text())
    expected_role = args.role if args.local else "p0"
    addresses = json.loads(
        subprocess.check_output(["ip", "-j", "-4", "addr"], text=True)
    )
    local_ips = {
        address["local"] for link in addresses for address in link["addr_info"]
    }
    if expected_role and cfg["nodes"][expected_role] not in local_ips:
        parser.error(
            f"Run this command on {expected_role}: {cfg['nodes'][expected_role]}"
        )
    if args.local:
        if not args.role or args.kind not in (
            "engine",
            "monitor",
            "proxy",
            "test",
            "production",
            "collective",
        ):
            parser.error("Local command needs --role and a singular job kind")
        local(cfg, args)
        return
    if args.action.startswith("gpu-") and args.kind != "engines":
        parser.error("Use gpu-check/gpu-clean engines")
    if args.pids and (args.action != "gpu-clean" or not args.only_role):
        parser.error("Explicit PIDs require gpu-clean engines --only-role NODE")
    if args.action == "worker" or args.kind in ("engine", "monitor", "collective"):
        parser.error("Use engines/monitors for cluster operations")
    if args.action == "start" and args.kind == "all":
        parser.error("Start monitors, engines, proxy and test explicitly in that order")
    cfg["expected_commit"] = subprocess.check_output(
        ["git", "-C", cfg["repo"], "rev-parse", "HEAD"], text=True
    ).strip()
    if args.kind == "production":
        cfg["production_run_id"] = args.run_id
    ssh = [
        "ssh",
        "-p",
        str(cfg["ssh_port"]),
        "-i",
        os.path.expanduser(cfg["ssh_key"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]
    remote_script = (
        f"{cfg['repo']}/examples/disaggregated/glm5next_int8/assessment_jobs.py"
    )
    groups = (
        ("production", "communication", "test", "proxy", "engines", "monitors")
        if args.kind == "all"
        else (args.kind,)
    )
    errors = []
    if args.action == "collect":
        destination = Path(cfg["log_root"]) / args.run_id / "collected"
        destination.mkdir(parents=True, exist_ok=True)
        for role, host in cfg["nodes"].items():
            if role == "p0":
                continue
            source = f"{cfg['log_root']}/{args.run_id}/{role}"
            with (destination / f"{role}.tar").open("wb") as out:
                result = subprocess.run(
                    ssh
                    + [
                        f"{cfg['ssh_user']}@{host}",
                        shlex.join(["tar", "-C", source, "-cf", "-", "."]),
                    ],
                    stdout=out,
                )
            if result.returncode:
                errors.append(role)
        print(f"Archives: {destination}; P0 data: {destination.parent / 'p0'}")
    else:
        phases = (
            ("gpu-check", "start")
            if args.action == "start" and args.kind in ("engines", "communication")
            else (args.action,)
        )
        for phase in phases:
            for group in groups:
                kind, roles = KINDS[group]
                for role in roles:
                    if args.only_role and role != args.only_role:
                        continue
                    cmd = [
                        f"{cfg['repo']}/.venv/bin/python",
                        remote_script,
                        phase,
                        kind,
                        "--local",
                        "--role",
                        role,
                        "--run-id",
                        args.run_id,
                        "--config",
                        "-",
                    ]
                    if args.pids:
                        cmd.extend(["--pids", *map(str, args.pids)])
                    if role == "p0":
                        target = cmd
                    else:
                        target = ssh + [
                            f"{cfg['ssh_user']}@{cfg['nodes'][role]}",
                            shlex.join(cmd),
                        ]
                    result = subprocess.run(target, input=json.dumps(cfg), text=True)
                    if result.returncode:
                        errors.append(f"{role}/{kind}")
            if errors:
                break
    if args.action == "stop" and "production" in groups:
        active = Path(cfg["log_root"]) / args.run_id / "p0/production/data/active.json"
        if active.exists():
            from production_assessment import control

            entry = json.loads(active.read_text())
            # Run outside the stopped supervisor's process group. Cleanup may
            # exceed its TERM grace period when a remote engine is hung.
            for kind in ("proxy", "engines", "monitors"):
                result = control(
                    entry["config"], entry["run_id"], "stop", kind, check=False
                )
                if result.returncode:
                    errors.append(f"production/{kind}")
    if errors:
        raise SystemExit(f"Failed targets: {errors}")


if __name__ == "__main__":
    # '-' permits one controller to pass an identical config to all nodes.
    if "--config" in sys.argv and sys.argv[sys.argv.index("--config") + 1] == "-":
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config:
            config.write(sys.stdin.read())
            config.flush()
            sys.argv[sys.argv.index("--config") + 1] = config.name
            main()
    else:
        main()
