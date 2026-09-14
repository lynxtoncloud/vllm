# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PD functional acceptance followed by streaming, end-to-end load tests."""

import argparse
import asyncio
import io
import json
import math
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pybase64 as base64
import regex as re
from PIL import Image

LENGTHS = {
    "text2k": 2047,
    "text32k": 32767,
    "text128k": 131071,
    "text512k": 524287,
    "text1m": 1047551,
}
MEDIA_CASES = ("image", "images", "video_frames", "video_mp4", "mixed")
ALL_CASES = (*LENGTHS, *MEDIA_CASES)
PERF_CASES = (*ALL_CASES, "mixed_load")
METRICS = (
    "vllm:nixl_bytes_transferred_sum",
    "vllm:nixl_bytes_transferred_count",
    "vllm:nixl_num_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total",
)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def long_prompt(tokenizer, length, nonce):
    """Build an exact token budget with three independently checked needles."""
    marker = "__PD_BODY_PLACEHOLDER__"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    prefix, suffix = template.split(marker)
    expected = {key: f"{nonce}-{key}" for key in ("start", "middle", "end")}

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    head = encode(
        prefix + f"Request ID: {nonce}.\n"
        "Read the document and return ONLY a JSON object with keys start, "
        "middle, end containing the corresponding secret values.\n"
        f"START_SECRET={expected['start']}\n"
    )
    middle = encode(f"\nMIDDLE_SECRET={expected['middle']}\n")
    tail = encode(
        f"\nEND_SECRET={expected['end']}\n"
        "Return the three secret values as JSON. No explanation.\n" + suffix
    )
    filler = encode(" Background record: ordinary information, no secret here.\n")
    remaining = length - len(head) - len(middle) - len(tail)
    if remaining < 0 or not filler:
        raise ValueError("Token budget is too small for the retrieval fixture")

    def pad(size):
        return (filler * math.ceil(size / len(filler)))[:size]

    ids = head + pad(remaining // 2) + middle
    ids += pad(remaining - remaining // 2) + tail
    assert len(ids) == length
    return ids, expected


def make_media(directory, nonce=""):
    """Create known-answer fixtures locally; no public URLs or shared mounts."""
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, color in enumerate(("red", "green", "blue")):
        frame = Image.new("RGB", (448, 448), color)
        # A tiny unique corner preserves the color answer while changing the
        # decoded pixels, preventing multimodal processor-cache reuse in perf.
        for column, value in enumerate(bytes.fromhex(nonce)):
            frame.paste(
                (value, 255 - value, value),
                (column * 4, 0, column * 4 + 4, 4),
            )
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG")
        frames.append(base64.b64encode(buffer.getvalue()).decode())
        frame.save(directory / f"{color}.png")
        for repeat in range(3):
            frame.save(directory / f"frame-{index * 3 + repeat:02d}.png")
    video = directory / "colors.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "1",
            "-i",
            str(directory / "frame-%02d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    return {
        "red": "data:image/jpeg;base64," + frames[0],
        "blue": "data:image/jpeg;base64," + frames[2],
        "frames": "data:video/jpeg;base64," + ",".join(frames),
        "mp4": "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode(),
    }


def media_prompt(case, media, nonce):
    def item(kind, url):
        return {"type": kind, kind: {"url": url}}

    if case == "image":
        content = [item("image_url", media["red"])]
        expected = {"color": "red"}
        question = 'What is the image color? Return JSON {"color":"..."}.'
    elif case == "images":
        content = [item("image_url", media[c]) for c in ("red", "blue")]
        expected = {"first": "red", "second": "blue"}
        question = 'Name the two image colors in order: {"first":"...","second":"..."}.'
    elif case in ("video_frames", "video_mp4"):
        content = [
            item("video_url", media["frames" if case == "video_frames" else "mp4"])
        ]
        expected = {"colors": ["red", "green", "blue"]}
        question = (
            'List video colors in time order, removing repeats: {"colors":[...]}.'
        )
    else:
        content = [item("image_url", media["blue"]), item("video_url", media["mp4"])]
        expected = {"image": "blue", "video": ["red", "green", "blue"]}
        question = (
            "Name the image color and video colors in time order, removing repeats: "
            '{"image":"...","video":[...]}.'
        )
    content.insert(
        0,
        {
            "type": "text",
            "text": f"Request ID: {nonce}. "
            + question
            + " Use lowercase English color names. Return ONLY JSON.",
        },
    )
    return [{"role": "user", "content": content}], expected


def matches_answer(text, expected):
    for candidate in re.findall(r"\{[^{}]*\}", text):
        try:
            if json.loads(candidate) == expected:
                return True
        except json.JSONDecodeError:
            pass
    return False


async def request(client, base_url, endpoint, payload, timeout):
    """Measure through the proxy, including P queue, prefill, transfer and D."""
    start = time.perf_counter()
    first = last = None
    text = ""
    usage = None
    finished = False
    done = False
    result = {"success": False}

    async def consume():
        nonlocal first, last, text, usage, finished, done
        async with client.stream("POST", base_url + endpoint, json=payload) as resp:
            if resp.is_error:
                body = (await resp.aread()).decode(errors="replace")
                raise RuntimeError(f"HTTP {resp.status_code}: {body[:2000]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    if line.startswith("{"):
                        raise RuntimeError(
                            f"Unexpected non-SSE response: {line[:2000]}"
                        )
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = choice.get("text") or delta.get("content") or ""
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                    if content or reasoning:
                        last = time.perf_counter()
                        if first is None:
                            first = last
                    text += content
                    finished |= choice.get("finish_reason") is not None

    try:
        await asyncio.wait_for(consume(), timeout=timeout)
        if not done or not finished or first is None or not text or not usage:
            raise RuntimeError("Incomplete SSE response, missing text, finish or usage")
        inputs, outputs = usage["prompt_tokens"], usage["completion_tokens"]
        if inputs <= 0 or outputs <= 0:
            raise RuntimeError(f"Invalid token usage: {usage}")
        result.update(
            success=True,
            input_tokens=inputs,
            output_tokens=outputs,
            ttft_s=first - start,
            tpot_s=(last - first) / (outputs - 1) if outputs > 1 else None,
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result.update(e2e_s=time.perf_counter() - start, text=text)
    return result


def metric_values(text):
    values = {}
    for line in text.splitlines():
        for name in METRICS:
            if line.startswith((name + "{", name + " ")):
                value = float(line.split()[1])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite metric: {line}")
                values[name] = values.get(name, 0) + value
    return values


def transfer_ok(before, after):
    # Missing samples and counter resets must not produce a false pass.
    if any(name not in before or name not in after for name in METRICS):
        return False
    return all(after[name] > before[name] for name in METRICS[:2]) and all(
        after[name] == before[name] for name in METRICS[2:]
    )


async def snapshot(client, url, path):
    response = await client.get(url + "/metrics")
    response.raise_for_status()
    path.write_text(response.text)
    return metric_values(response.text)


def summarize(rows, duration):
    good = [r for r in rows if r["success"]]
    result = {
        "requests": len(rows),
        "completed": len(good),
        "failed": len(rows) - len(good),
        "duration_s": duration,
    }
    for name in ("input_tokens", "output_tokens"):
        result[name] = sum(r[name] for r in good)
        result[name + "_per_s"] = result[name] / duration
    result["requests_per_s"] = len(good) / duration
    for metric in ("ttft_s", "tpot_s", "e2e_s"):
        values = sorted(r[metric] for r in good if r[metric] is not None)
        result[metric] = {
            f"p{p}": values[math.ceil(len(values) * p / 100) - 1] if values else None
            for p in (50, 95, 99)
        }
    return result


def accepted(report, config):
    return (
        report.get("passed") is True
        and report.get("config") == config
        and [s["case"] for s in report.get("stages", [])] == list(ALL_CASES)
        and all(s["failed"] == 0 and s["transfer_ok"] for s in report["stages"])
    )


async def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model,
        "tokenizer": str(args.tokenizer),
        "base_url": args.base_url,
        "prefill_url": args.prefill_url,
        "decode_url": args.decode_url,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    report_path = args.output_dir / "functional.json"
    if args.mode == "perf":
        report = json.loads(report_path.read_text())
        if not accepted(report, config):
            raise ValueError(
                "Functional acceptance must pass with the same configuration"
            )
        save(
            args.output_dir / "performance.json",
            {"config": config, "passed": False, "stages": []},
        )
    else:
        save(report_path, {"config": config, "passed": False, "stages": []})

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    media = make_media(args.output_dir / "assets")
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(args.timeout, connect=30),
        limits=limits,
        trust_env=False,
    ) as client:
        for url, route in (
            (args.base_url, "/healthcheck"),
            (args.prefill_url, "/health"),
            (args.decode_url, "/health"),
        ):
            response = await client.get(url + route)
            response.raise_for_status()

        stages = []
        cases = ALL_CASES if args.mode == "functional" else args.cases.split(",")
        levels = [1] if args.mode == "functional" else args.concurrency
        for case in cases:
            for concurrency in levels:
                label = f"{args.mode}-{case}-c{concurrency}-{time.time_ns()}"
                count = (
                    1
                    if args.mode == "functional"
                    else max(args.requests, concurrency * 4)
                )
                # Prepare outside the timed interval, with distinct prefixes to
                # avoid measuring repeated prompt-cache hits across requests.
                prepared = []
                for index in range(count):
                    request_case = (
                        ("text2k", "text32k", "image", "images", "video_mp4", "mixed")[
                            index % 6
                        ]
                        if case == "mixed_load"
                        else case
                    )
                    nonce = uuid.uuid4().hex
                    payload = {
                        "model": args.model,
                        "temperature": 0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "max_tokens": 1024
                        if args.mode == "functional"
                        else args.output_tokens,
                    }
                    if request_case in LENGTHS:
                        ids, expected = long_prompt(
                            tokenizer, LENGTHS[request_case], nonce
                        )
                        payload.update(prompt=ids, add_special_tokens=False)
                        endpoint = "/v1/completions"
                    else:
                        request_media = (
                            make_media(
                                args.output_dir / "assets" / label / str(index), nonce
                            )
                            if args.mode == "perf"
                            else media
                        )
                        messages, expected = media_prompt(
                            request_case, request_media, nonce
                        )
                        payload.update(
                            messages=messages,
                            reasoning_effort="low",
                        )
                        endpoint = "/v1/chat/completions"
                    if args.mode == "perf":
                        payload["ignore_eos"] = True
                    prepared.append((endpoint, payload, expected, request_case))

                before = await snapshot(
                    client, args.decode_url, args.output_dir / f"{label}-D-before.prom"
                )
                await snapshot(
                    client, args.prefill_url, args.output_dir / f"{label}-P-before.prom"
                )
                sem = asyncio.Semaphore(concurrency)

                async def one(index, entry, sem=sem, label=label):
                    endpoint, payload, expected, request_case = entry
                    async with sem:
                        row = await request(
                            client, args.base_url, endpoint, payload, args.timeout
                        )
                    row.update(index=index, case=request_case)
                    if (
                        row["success"]
                        and request_case in LENGTHS
                        and row["input_tokens"] != LENGTHS[request_case]
                    ):
                        row.update(
                            success=False,
                            error="Server prompt token count differs from fixture",
                        )
                    if args.mode == "functional":
                        row["expected"] = expected
                        if row["success"] and not matches_answer(row["text"], expected):
                            row.update(
                                success=False,
                                error="Answer did not match known fixture",
                            )
                    elif row["success"] and row["output_tokens"] != args.output_tokens:
                        row.update(
                            success=False,
                            error="Output did not reach requested token budget",
                        )
                    status = "PASS" if row["success"] else "FAIL"
                    print(f"{label} request {index}: {status}", flush=True)
                    # Each completed response survives interruption of later requests.
                    with (args.output_dir / f"{label}.jsonl").open("a") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    return row

                print(f"Starting {label}: {count} requests", flush=True)
                started = time.perf_counter()
                rows = await asyncio.gather(
                    *(one(i, entry) for i, entry in enumerate(prepared))
                )
                duration = time.perf_counter() - started
                # Allow asynchronous connector metrics publication after the requests.
                for _ in range(10):
                    await asyncio.sleep(2)
                    after = await snapshot(
                        client,
                        args.decode_url,
                        args.output_dir / f"{label}-D-after.prom",
                    )
                    if transfer_ok(before, after):
                        break
                await snapshot(
                    client, args.prefill_url, args.output_dir / f"{label}-P-after.prom"
                )
                stage = summarize(rows, duration)
                stage.update(
                    case=case,
                    concurrency=concurrency,
                    label=label,
                    transfer_ok=transfer_ok(before, after),
                )
                stages.append(stage)
                save(args.output_dir / f"{label}-summary.json", stage)
                passed = all(s["failed"] == 0 and s["transfer_ok"] for s in stages)
                complete = len(stages) == len(cases) * len(levels)
                report = {
                    "config": config,
                    "passed": passed and complete,
                    "stages": stages,
                }
                save(
                    report_path
                    if args.mode == "functional"
                    else args.output_dir / "performance.json",
                    report,
                )
                print(json.dumps(stage, ensure_ascii=False), flush=True)
                if not passed:
                    raise RuntimeError(
                        f"Stopped at {label}; inspect response JSONL and metrics"
                    )
        print(f"{args.mode} complete: {args.output_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("functional", "perf"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefill-url", default="http://10.5.10.36:8001")
    parser.add_argument("--decode-url", default="http://10.5.10.55:8002")
    parser.add_argument("--model", default="glm53-int8")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/data/models/zai-org/GLM-5.3-Flash-W8A16-G128"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(PERF_CASES))
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=16,
        help="Minimum requests per level; at least 4x concurrency",
    )
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument(
        "--timeout",
        type=float,
        default=14400,
        help="Per-request wall-clock timeout in seconds",
    )
    args = parser.parse_args()
    if any(c not in PERF_CASES for c in args.cases.split(",")):
        parser.error(f"--cases must select from {PERF_CASES}")
    if min(args.concurrency) < 1 or args.requests < 1 or args.timeout <= 0:
        parser.error("Concurrency, requests and timeout must be positive")
    if not 1 <= args.output_tokens <= 1024:
        parser.error("--output-tokens must be 1..1024 to fit the 1M fixture")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
