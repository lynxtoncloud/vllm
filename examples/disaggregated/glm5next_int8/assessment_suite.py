# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resumable PD functional and performance matrix, including 1M and media."""

import argparse
import asyncio
import io
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pybase64 as base64
from PIL import Image, ImageDraw
from run_checks import (
    LENGTHS,
    MEDIA_CASES,
    long_prompt,
    make_media,
    matches_answer,
    media_prompt,
    request,
    snapshot,
    summarize,
    transfer_ok,
)

CASES = (
    "text2k",
    *MEDIA_CASES,
    "image_large",
    "ocr",
    "chart",
    "text_image32k",
    "text32k",
    "text128k",
    "text512k",
    "text1m",
    "mixed_load",
)


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def image_case(case, media, nonce):
    if case == "image_large":
        image = Image.new("RGB", (1344, 1344), "red")
        expected, question = (
            {"color": "red"},
            'Return JSON {"color":"..."} for the image color.',
        )
    elif case == "ocr":
        image = Image.new("RGB", (1024, 512), "white")
        expected = {"code": "PD-" + nonce[:8].upper()}
        ImageDraw.Draw(image).text(
            (50, 180), expected["code"], fill="black", font_size=64
        )
        question = 'Read the printed code. Return ONLY JSON {"code":"..."}.'
    else:
        image = Image.new("RGB", (1024, 768), "white")
        draw = ImageDraw.Draw(image)
        for x, height, label in ((100, 120, "A"), (400, 400, "B"), (700, 240, "C")):
            draw.rectangle((x, 620 - height, x + 130, 620), fill="blue")
            draw.text((x + 40, 635), label, fill="black", font_size=48)
        expected, question = (
            {"largest": "B"},
            'Which bar is tallest? Return JSON {"largest":"..."}.',
        )
    # Unique pixels, not just unique transport encodings.
    image.putpixel((0, 0), tuple(bytes.fromhex(nonce[:6])))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Request {nonce}. {question}"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ], expected


def prepare(case, mode, cfg, tokenizer, directory):
    nonce = uuid.uuid4().hex
    payload = {
        "model": cfg["model_name"],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 1024 if mode == "functional" else cfg["output_tokens"],
    }
    if mode == "perf":
        payload["ignore_eos"] = True
    if case in LENGTHS:
        ids, expected = long_prompt(tokenizer, LENGTHS[case], nonce)
        payload.update(prompt=ids, add_special_tokens=False)
        return "/v1/completions", payload, expected
    if case in ("image_large", "ocr", "chart"):
        messages, expected = image_case(case, None, nonce)
    else:
        media = make_media(directory, nonce)
        messages, expected = media_prompt(
            "image" if case == "text_image32k" else case, media, nonce
        )
        if case == "text_image32k":
            filler = tokenizer.encode(
                " ordinary background record.", add_special_tokens=False
            )
            ids = (filler * (32768 // len(filler) + 1))[:32768]
            # 32K text body plus image/template tokens; actual billed input is recorded.
            messages[0]["content"][0]["text"] += "\n" + tokenizer.decode(ids)
            messages[0]["content"].append(
                {
                    "type": "text",
                    "text": 'Return JSON {"color":"..."} for the image color.',
                }
            )
        shutil.rmtree(directory)
    payload.update(messages=messages, reasoning_effort="low")
    return "/v1/chat/completions", payload, expected


def plan(cfg):
    for name in (
        "minimum_requests",
        "functional_repeats",
        "output_tokens",
        "request_timeout",
    ):
        if cfg[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg["output_tokens"] > 1024 or cfg["warmups"] < 0:
        raise ValueError("output_tokens must be <=1024; warmups must be >=0")
    if not cfg["concurrency"] or any(
        not isinstance(c, int) or c < 1 for c in cfg["concurrency"]
    ):
        raise ValueError("Concurrency levels must be positive integers")
    cases = cfg.get("cases", list(CASES))
    if not cases or any(case not in CASES for case in cases):
        raise ValueError(f"Unknown/empty cases; supported: {CASES}")
    return [("functional", case, 1) for case in cases if case != "mixed_load"] + [
        ("perf", case, c) for case in cases for c in cfg["concurrency"]
    ]


def stage_passed(data):
    return bool(
        data.get("passed")
        and data.get("completed") == data.get("expected_requests")
        and data.get("failed") == 0
        and data.get("transfer_ok")
    )


async def health(client, urls):
    for url, route in (
        (urls[0], "/healthcheck"),
        (urls[1], "/health"),
        (urls[2], "/health"),
    ):
        response = await client.get(url + route, timeout=30)
        response.raise_for_status()


async def run(cfg, out):
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    matrix = plan(cfg)
    metadata = {
        "config": cfg,
        "commit": subprocess.check_output(
            ["git", "-C", cfg["repo"], "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Config/code changed. Start a new run ID, do not mix results.")
    save(manifest, metadata)
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "Install ffmpeg (including libx264) before starting the suite"
        )
    # Verify codec before starting any long request.
    make_media(out / "fixtures")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    urls = (
        f"http://{cfg['nodes']['p0']}:8000",
        f"http://{cfg['nodes']['p0']}:8001",
        f"http://{cfg['nodes']['d0']}:8002",
    )
    async with httpx.AsyncClient(
        trust_env=False,
        timeout=30,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    ) as client:
        await health(client, urls)
        for url in urls[1:]:
            response = await client.post(
                url + "/tokenize",
                json={"model": cfg["model_name"], "prompt": "context check"},
            )
            response.raise_for_status()
            if response.json()["max_model_len"] < 1048576:
                raise RuntimeError(f"{url} does not have 1M context configured")
        failures = []
        for index, (mode, case, concurrency) in enumerate(matrix):
            key = f"{mode}-{case}-c{concurrency}"
            stage_dir = out / key
            stage_dir.mkdir(exist_ok=True)
            result_file = stage_dir / "result.json"
            if result_file.exists() and stage_passed(
                json.loads(result_file.read_text())
            ):
                print(f"RESUME skip passed {key}", flush=True)
                continue
            if mode == "perf":
                dependencies = (
                    ("text2k", "image", "video_mp4")
                    if case == "mixed_load"
                    else (case,)
                )
                if not all(
                    (out / f"functional-{d}-c1/result.json").exists()
                    and stage_passed(
                        json.loads((out / f"functional-{d}-c1/result.json").read_text())
                    )
                    for d in dependencies
                ):
                    save(
                        result_file,
                        {"passed": False, "status": "blocked_functional", "case": case},
                    )
                    failures.append(key)
                    continue
            await health(client, urls)
            attempt = stage_dir / str(time.time_ns())
            attempt.mkdir()
            save(
                out / "progress.json",
                {
                    "status": "running",
                    "stage": key,
                    "stage_number": index + 1,
                    "total_stages": len(matrix),
                    "time": time.time(),
                    "attempt": str(attempt),
                },
            )
            count = (
                cfg["functional_repeats"]
                if mode == "functional"
                else max(cfg["minimum_requests"], concurrency * 4)
            )
            entries = []
            for i in range(count + (cfg["warmups"] if mode == "perf" else 0)):
                actual = (
                    ("text2k", "image", "video_mp4")[i % 3]
                    if case == "mixed_load"
                    else case
                )
                entries.append(
                    (
                        actual,
                        prepare(actual, mode, cfg, tokenizer, attempt / f"media-{i}"),
                    )
                )

            async def one(i, entry, mode=mode, attempt=attempt, key=key):
                actual, (endpoint, payload, expected) = entry
                row = await request(
                    client, urls[0], endpoint, payload, cfg["request_timeout"]
                )
                row.update(
                    index=i, case=actual, finished=time.time(), expected=expected
                )
                if (
                    row["success"]
                    and actual in LENGTHS
                    and row["input_tokens"] != LENGTHS[actual]
                ):
                    row.update(success=False, error="Incorrect input token budget")
                row["answer_matches"] = matches_answer(row["text"], expected)
                if (
                    mode == "functional"
                    and row["success"]
                    and not row["answer_matches"]
                ):
                    row.update(
                        success=False, error="Known answer mismatch", expected=expected
                    )
                if (
                    mode == "perf"
                    and row["success"]
                    and row["output_tokens"] != cfg["output_tokens"]
                ):
                    row.update(success=False, error="Incorrect output token budget")
                with (attempt / "requests.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"{key} request {i}: {row['success']} {row.get('error', '')}",
                    flush=True,
                )
                return row

            if mode == "perf":
                for i, entry in enumerate(entries[count:]):
                    warmup = await one(-i - 1, entry)
                    if not warmup["success"]:
                        raise RuntimeError(f"Warmup failed: {key}; see {attempt}")
            before = await snapshot(client, urls[2], attempt / "D-before.prom")
            await snapshot(client, urls[1], attempt / "P-before.prom")
            sem = asyncio.Semaphore(concurrency)

            async def limited(i, entry, sem=sem, one=one):
                async with sem:
                    return await one(i, entry)

            started = time.time()
            clock = time.perf_counter()
            tasks = [
                asyncio.create_task(limited(i, e))
                for i, e in enumerate(entries[:count])
            ]
            batch = asyncio.gather(*tasks)

            async def watch_health():
                while True:
                    await asyncio.sleep(30)
                    await health(client, urls)

            watchdog = asyncio.create_task(watch_health())
            done, _ = await asyncio.wait(
                (batch, watchdog), return_when=asyncio.FIRST_COMPLETED
            )
            interrupted = watchdog in done
            watchdog.cancel()
            if interrupted:
                batch.cancel()
            await asyncio.gather(watchdog, batch, return_exceptions=True)
            rows = [
                task.result()
                for task in tasks
                if not task.cancelled() and task.exception() is None
            ]
            duration, ended = time.perf_counter() - clock, time.time()
            stage = summarize(rows, duration)
            stage.update(
                mode=mode,
                case=case,
                concurrency=concurrency,
                started=started,
                ended=ended,
                expected_requests=count,
                attempt=str(attempt),
            )
            for field in ("ttft_s", "tpot_s", "e2e_s"):
                values = [
                    r[field] for r in rows if r["success"] and r[field] is not None
                ]
                stage[field]["mean"] = sum(values) / len(values) if values else None
            # Persist request results even if the server disappears before metrics.
            stage.update(
                passed=False,
                transfer_ok=False,
                interrupted=interrupted,
                unfinished=count - len(rows),
            )
            save(result_file, stage)
            if interrupted:
                raise RuntimeError(
                    f"Server health failed at {key}; partial results saved"
                )
            try:
                for _ in range(10):
                    await asyncio.sleep(2)
                    after = await snapshot(client, urls[2], attempt / "D-after.prom")
                    if transfer_ok(before, after):
                        break
                await snapshot(client, urls[1], attempt / "P-after.prom")
                stage["transfer_ok"] = transfer_ok(before, after)
            except Exception as error:
                stage["metrics_error"] = str(error)
            stage["passed"] = (
                stage["failed"] == 0
                and stage["unfinished"] == 0
                and stage["transfer_ok"]
            )
            save(attempt / "result.json", stage)
            save(result_file, stage)
            if not stage["passed"]:
                failures.append(key)
                if mode == "perf":
                    raise RuntimeError(
                        f"Performance stage failed: {key}; resume after recovery"
                    )
            await health(client, urls)
        save(
            out / "progress.json",
            {
                "status": "complete" if not failures else "complete_with_failures",
                "failed_stages": failures,
                "time": time.time(),
                "total_stages": len(matrix),
            },
        )
        if failures:
            raise RuntimeError(f"Failed/blocked stages: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    try:
        asyncio.run(run(cfg, args.output))
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "progress.json"
        progress = json.loads(path.read_text()) if path.exists() else {}
        progress.update(status="stopped_with_error", error=str(error), time=time.time())
        save(path, progress)
        raise


if __name__ == "__main__":
    main()
