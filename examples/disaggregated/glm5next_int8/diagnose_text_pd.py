# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the same retrieval prompt through local P, local D and PD transfer."""

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

import httpx
from assessment_suite import save
from production_assessment import wait_ready
from run_checks import long_prompt, matches_answer, request, snapshot, transfer_ok


def prepare_input(tokenizer, length, nonce, out):
    ids, expected = long_prompt(tokenizer, length, nonce)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    markers = {
        key: {"count": decoded.count(value), "character_offset": decoded.find(value)}
        for key, value in expected.items()
    }
    encoded = json.dumps(ids).encode()
    audit = {
        "input_tokens": len(ids),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "expected": expected,
        "markers": markers,
    }
    (out / "prompt-ids.json").write_bytes(encoded)
    (out / "prompt.txt").write_text(decoded)
    save(out / "input-audit.json", audit)
    if len(ids) != length or any(item["count"] != 1 for item in markers.values()):
        raise ValueError(
            "Retrieval fixture lost/duplicated a marker; see input-audit.json"
        )
    return ids, audit


async def compare(cfg, ids, audit, out):
    routes = (
        ("p", f"http://{cfg['nodes']['p0']}:8001"),
        ("d", f"http://{cfg['nodes']['d0']}:8002"),
        ("pd", f"http://{cfg['nodes']['p0']}:8000"),
    )
    results = []
    async with httpx.AsyncClient(trust_env=False) as client:
        for name, url in routes:
            payload = {
                "model": cfg["model_name"],
                "prompt": ids,
                "add_special_tokens": False,
                "temperature": 0,
                "max_tokens": 1024,
                "stream": True,
                "stream_options": {"include_usage": True},
                # Keep identical model input, but avoid prefix reuse across routes.
                "cache_salt": uuid.uuid4().hex,
            }
            save(out / f"{name}-request.json", payload)
            save(out / "progress.json", {"status": "running", "route": name})
            print(f"Starting {name}: {url}", flush=True)
            before = None
            if name == "pd":
                before = await snapshot(client, routes[1][1], out / "D-before.prom")
            row = await request(client, url, "/v1/completions", payload, 1800)
            row.update(
                route=name,
                prompt_sha256=audit["sha256"],
                expected=audit["expected"],
                answer_matches=matches_answer(row["text"], audit["expected"]),
            )
            row["passed"] = bool(
                row["success"]
                and row["answer_matches"]
                and row["input_tokens"] == len(ids)
            )
            save(out / f"{name}-response.json", row)
            if name == "pd":
                try:
                    for _ in range(10):
                        await asyncio.sleep(2)
                        after = await snapshot(
                            client, routes[1][1], out / "D-after.prom"
                        )
                        if transfer_ok(before, after):
                            break
                    row["transfer_ok"] = transfer_ok(before, after)
                    row["passed"] &= row["transfer_ok"]
                except Exception as error:
                    row.update(passed=False, metrics_error=str(error))
                save(out / f"{name}-response.json", row)
            results.append(row)
            save(out / "results.json", results)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    save(
        out / "progress.json",
        {"status": "completed", "all_passed": all(row["passed"] for row in results)},
    )


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=32767)
    parser.add_argument("--nonce", default="99d009d99818401ca25a163f63381b1d")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    save(args.output / "config.json", cfg)
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
        ids, audit = prepare_input(tokenizer, args.length, args.nonce, args.output)
        print(json.dumps(audit, ensure_ascii=False), flush=True)
        save(args.output / "progress.json", {"status": "waiting_for_services"})
        asyncio.run(wait_ready(cfg, 3600))
        asyncio.run(compare(cfg, ids, audit, args.output))
    except Exception as error:
        save(
            args.output / "progress.json",
            {"status": "stopped_with_error", "error": str(error), "time": time.time()},
        )
        raise


if __name__ == "__main__":
    main()
