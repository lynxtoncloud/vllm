# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect VRAM owners and release verified orphan workers using Linux pidfds."""

import json
import os
import select
import signal
import time
from contextlib import suppress
from pathlib import Path

from assessment_monitor import gpu_samples


def process(pid):
    from assessment_jobs import identity

    root = Path(f"/proc/{pid}")
    before = identity(pid)
    if not before:
        return None
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        devices = set()
        for fd in (root / "fd").iterdir():
            try:
                target = os.readlink(fd)
            except FileNotFoundError:
                continue
            if target == "/dev/kfd" or target.startswith("/dev/dri/"):
                devices.add(target)
        result = dict(
            before,
            pid=pid,
            ppid=int(fields[1]),
            comm=(root / "comm").read_text().strip(),
            cwd=os.readlink(root / "cwd"),
            devices=sorted(devices),
        )
        after = identity(pid)
        if not after or any(
            after[k] != before[k] for k in ("start_ticks", "boot_id", "session")
        ):
            return None
        return result
    except (OSError, ValueError):
        return None


def eligible(proc, cfg, recorded_sessions, explicit=False):
    if not proc or proc["state"] == "Z" or not proc["devices"]:
        return False
    if proc["ppid"] != 1 or not proc["comm"].startswith("VLLM::Worker"):
        return False
    if Path(proc["cwd"]).resolve() != Path(cfg["repo"]).resolve():
        return False
    from assessment_jobs import identity

    if explicit:
        # Explicit legacy PIDs still cannot target a running managed engine.
        for record in recorded_sessions:
            if (
                record.get("boot_id") == proc["boot_id"]
                and record["pid"] == proc["session"]
            ):
                leader = identity(record["pid"])
                if leader and leader["state"] != "Z":
                    return False
        return True
    for record in recorded_sessions:
        leader = identity(record["pid"])
        if (
            record.get("boot_id") == proc["boot_id"]
            and record["pid"] == proc["session"]
            and int(proc["start_ticks"]) >= int(record["start_ticks"])
            and (leader is None or leader["state"] == "Z")
        ):
            return True
    return False


def inspect(cfg, role):
    records = []
    for path in Path(cfg["log_root"]).glob(f"*/{role}/engine/state.json"):
        with suppress(OSError, ValueError):
            records.append(json.loads(path.read_text()))
    owners = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            proc = process(int(entry.name))
            if proc and proc["devices"]:
                proc["tracked_orphan"] = eligible(proc, cfg, records)
                proc["legacy_orphan"] = eligible(proc, cfg, records, explicit=True)
                owners.append(proc)
    cards = []
    utilization = float(cfg["environment"].get("GPU_MEMORY_UTILIZATION", 0.90))
    if not 0 < utilization <= 1:
        raise ValueError("GPU_MEMORY_UTILIZATION must be in (0, 1]")
    for name, gpu in gpu_samples().items():
        total, used = gpu["mem_info_vram_total"], gpu["mem_info_vram_used"]
        valid = (
            total is not None and used is not None and total > 0 and 0 <= used <= total
        )
        free = total - used if valid else None
        required = total * utilization if valid else None
        cards.append(
            {
                "card": name,
                "pci": gpu["pci"],
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "required_free_bytes": required,
                "enough": valid and free >= required,
            }
        )
    return {
        "role": role,
        "time": time.time(),
        "cards": cards,
        "owners": owners,
        "ready": len(cards) == 8 and all(card["enough"] for card in cards),
        "note": "sysfs estimate; actual HIP free memory and profiling may differ",
    }


def release(proc, cfg, explicit=False):
    from assessment_jobs import identity

    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        return {"pid": proc["pid"], "error": "pidfd unavailable; no signal sent"}
    try:
        fd = os.pidfd_open(proc["pid"])
    except ProcessLookupError:
        return {"pid": proc["pid"], "result": "already_exited"}
    except OSError as error:
        return {"pid": proc["pid"], "error": str(error)}
    try:
        current = process(proc["pid"])
        same = current and all(
            current[k] == proc[k]
            for k in ("start_ticks", "boot_id", "session", "comm", "cwd")
        )
        leader = identity(proc["session"])
        if not same or not eligible(current, cfg, [], explicit=True):
            return {"pid": proc["pid"], "error": "identity/ownership changed; skipped"}
        if not explicit and leader and leader["state"] != "Z":
            return {"pid": proc["pid"], "error": "session leader is alive; skipped"}
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        if poll.poll(5000):
            return {"pid": proc["pid"], "result": "exited_after_TERM"}
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        return {
            "pid": proc["pid"],
            "result": "exited_after_KILL"
            if poll.poll(10000)
            else "still_present_after_KILL",
        }
    except ProcessLookupError:
        return {"pid": proc["pid"], "result": "already_exited"}
    except OSError as error:
        return {"pid": proc["pid"], "error": str(error)}
    finally:
        os.close(fd)


def run(cfg, role, action, pids, output):
    before = inspect(cfg, role)
    actions = []
    if action == "gpu-clean":
        owners = {owner["pid"]: owner for owner in before["owners"]}
        targets = (
            pids if pids else [p for p, v in owners.items() if v["tracked_orphan"]]
        )
        for pid in targets:
            proc = owners.get(pid)
            if proc is None or not (
                proc["tracked_orphan"] or (pids and proc["legacy_orphan"])
            ):
                actions.append(
                    {"pid": pid, "error": "not an eligible orphan GPU worker; skipped"}
                )
                continue
            actions.append(release(proc, cfg, explicit=bool(pids)))
        after = inspect(cfg, role)
        for _ in range(10):
            if after["ready"] or not actions:
                break
            time.sleep(1)
            after = inspect(cfg, role)
    else:
        after = before
    result = {"before": before, "actions": actions, "after": after}
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{action}-{time.time_ns()}.json").write_text(
        json.dumps(result, indent=2)
    )
    print(f"{role}: VRAM startup capacity ready={after['ready']}", flush=True)

    def gib(value):
        return "unknown" if value is None else f"{value / 2**30:.2f}"

    for card in after["cards"]:
        print(
            f"  {card['card']} PCI={card['pci']} used={gib(card['used_bytes'])} GiB "
            f"free={gib(card['free_bytes'])} GiB "
            f"required={gib(card['required_free_bytes'])} GiB"
        )
    for owner in after["owners"]:
        print(
            f"  PID={owner['pid']} PPID={owner['ppid']} {owner['comm']} "
            f"cwd={owner['cwd']} tracked_orphan={owner['tracked_orphan']} "
            f"legacy_orphan={owner['legacy_orphan']} devices={owner['devices']}"
        )
    for action_result in actions:
        print(json.dumps(action_result), flush=True)
    print(f"Details: {output}", flush=True)
    return after["ready"] and all(
        "error" not in item and item.get("result") != "still_present_after_KILL"
        for item in actions
    )
