# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the PD acceptance client, without a vLLM server."""

import asyncio
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/disaggregated/glm5next_int8/run_checks.py"
)
spec = importlib.util.spec_from_file_location("pd_checks", SCRIPT)
assert spec is not None and spec.loader is not None
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant>"

    def encode(self, text, **kwargs):
        return list(map(ord, text))


@pytest.mark.parametrize("length", checks.LENGTHS.values())
def test_retrieval_budget_preserves_all_needles_and_generation_suffix(length):
    ids, expected = checks.long_prompt(CharacterTokenizer(), length, "unique-test")
    text = "".join(map(chr, ids))
    assert len(ids) == length
    assert length % 4 != 0  # Exercise kpool's partial final pool.
    assert text.endswith("</user><assistant>")
    positions = [text.index(expected[k]) for k in ("start", "middle", "end")]
    assert positions[0] < length * 0.3
    assert length * 0.3 < positions[1] < length * 0.7
    assert positions[2] > length * 0.7
    assert all(text.count(value) == 1 for value in expected.values())


def test_answer_check_rejects_wrong_order_or_partial_retrieval():
    expected = {"colors": ["red", "green", "blue"]}
    assert checks.matches_answer(
        '```json\n{"colors":["red","green","blue"]}\n```', expected
    )
    assert not checks.matches_answer('{"colors":["blue","green","red"]}', expected)
    assert not checks.matches_answer("red green blue", expected)


@pytest.mark.parametrize("invalid", ["missing", "reset", "failure", "no-transfer"])
def test_transfer_check_requires_progress_without_failures(invalid):
    before = dict(zip(checks.METRICS, (100, 1, 0, 0)))
    after = dict(zip(checks.METRICS, (200, 2, 0, 0)))
    assert checks.transfer_ok(before, after)
    if invalid == "missing":
        del after[checks.METRICS[0]]
    elif invalid == "reset":
        after[checks.METRICS[0]] = 50
    elif invalid == "failure":
        after[checks.METRICS[2]] = 1
    else:
        after = before.copy()
    assert not checks.transfer_ok(before, after)


def test_metrics_sum_engines_but_exclude_histogram_buckets():
    name = checks.METRICS[0]
    assert (
        checks.metric_values(
            f'{name}{{engine="0"}} 10\n{name}{{engine="1"}} 20\n'
            'vllm:nixl_bytes_transferred_bucket{le="100"} 999\n'
        )[name]
        == 30
    )


def test_incomplete_functional_run_cannot_unlock_performance():
    config = {"model": "test"}
    stages = [
        {"case": case, "failed": 0, "transfer_ok": True} for case in checks.ALL_CASES
    ]
    report = {
        "config": config,
        "passed": True,
        "stages": stages,
    }
    assert checks.accepted(report, config)
    assert not checks.accepted(report, {"model": "different"})
    stages.pop()
    assert not checks.accepted(report, config)


def test_failed_requests_do_not_inflate_throughput():
    rows = [
        {
            "success": True,
            "input_tokens": 100,
            "output_tokens": 10,
            "ttft_s": 1,
            "tpot_s": 0.2,
            "e2e_s": 3,
        },
        {"success": False, "input_tokens": 900, "output_tokens": 90},
    ]
    result = checks.summarize(rows, 10)
    assert result["completed"] == 1 and result["failed"] == 1
    assert result["input_tokens_per_s"] == 10
    assert result["output_tokens_per_s"] == 1
    assert result["ttft_s"]["p99"] == 1


@pytest.mark.parametrize(
    "variant", ["ok", "truncated", "error", "no-usage", "http-error"]
)
def test_stream_accepts_only_complete_successful_responses(variant):
    events = [
        {"choices": [{"delta": {"content": "red"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    if variant != "no-usage":
        events.append(
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 2}}
        )
    if variant == "error":
        events.append({"error": {"message": "remote worker died"}})
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    if variant != "truncated":
        body += "data: [DONE]\n\n"

    async def execute():
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                503 if variant == "http-error" else 200,
                content=body,
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            return await checks.request(
                client, "http://test", "/v1/chat/completions", {}, 5
            )

    result = asyncio.run(execute())
    assert result["success"] == (variant == "ok")
    if variant == "ok":
        assert result["text"] == "red"
        assert result["input_tokens"] == 100
        assert result["e2e_s"] >= result["ttft_s"] >= 0


def test_mixed_fixture_sends_both_media_types_with_known_answer():
    media = {"red": "red-url", "blue": "blue-url", "mp4": "video-url"}
    messages, answer = checks.media_prompt("mixed", media, "test")
    assert [part["type"] for part in messages[0]["content"]] == [
        "text",
        "image_url",
        "video_url",
    ]
    assert answer == {"image": "blue", "video": ["red", "green", "blue"]}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_generated_mp4_preserves_color_order_and_unique_media(tmp_path):
    first = checks.make_media(tmp_path / "one", "0123456789abcdef")
    second = checks.make_media(tmp_path / "two", "fedcba9876543210")
    assert first["red"] != second["red"]
    assert first["mp4"] != second["mp4"]
    decoded = subprocess.check_output(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(tmp_path / "one/colors.mp4"),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    frame_size = 448 * 448 * 3
    assert len(decoded) == frame_size * 9
    for i in range(9):
        offset = i * frame_size + (224 * 448 + 224) * 3
        pixel = decoded[offset : offset + 3]
        assert max(range(3), key=lambda channel: pixel[channel]) == i // 3


def test_full_client_workflow_with_mock_http_and_performance_gate(
    tmp_path, monkeypatch
):
    """Exercise persisted acceptance and load-test orchestration, not model quality."""
    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *a, **kw: CharacterTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    media = dict.fromkeys(("red", "blue", "frames", "mp4"), "mock-media")
    monkeypatch.setattr(checks, "make_media", lambda *args: media)
    calls: list[dict[str, object]] = []

    async def no_sleep(_):
        pass

    monkeypatch.setattr(checks.asyncio, "sleep", no_sleep)

    def handle(req):
        if req.method == "GET":
            if req.url.path == "/metrics":
                values = (len(calls) * 100, len(calls), 0, 0)
                return httpx.Response(
                    200,
                    text="\n".join(
                        f"{key} {value}" for key, value in zip(checks.METRICS, values)
                    ),
                )
            return httpx.Response(200, json={"status": "ok"})
        payload = json.loads(req.content)
        calls.append(payload)
        if "prompt" in payload:
            prompt = "".join(map(chr, payload["prompt"]))
            answer = {
                key: checks.re.search(key.upper() + r"_SECRET=([^\s]+)", prompt).group(
                    1
                )
                for key in ("start", "middle", "end")
            }
            inputs = len(payload["prompt"])
        else:
            question = payload["messages"][0]["content"][0]["text"]
            inputs = 100
            if "two image" in question:
                answer = {"first": "red", "second": "blue"}
            elif "image color and video" in question:
                answer = {"image": "blue", "video": ["red", "green", "blue"]}
            elif "video colors" in question:
                answer = {"colors": ["red", "green", "blue"]}
            else:
                answer = {"color": "red"}
        event = {
            "choices": [{"text": json.dumps(answer), "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": inputs,
                "completion_tokens": payload["max_tokens"],
            },
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        checks.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(handle),
            **kwargs,
        ),
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        model="test",
        tokenizer=tmp_path,
        base_url="http://proxy",
        prefill_url="http://prefill",
        decode_url="http://decode",
        mode="functional",
        timeout=5,
        cases="mixed_load",
        concurrency=[1],
        requests=1,
        output_tokens=256,
    )
    asyncio.run(checks.run(args))
    report = json.loads((tmp_path / "functional.json").read_text())
    assert report["passed"] and len(report["stages"]) == len(checks.ALL_CASES)
    args.mode = "perf"
    asyncio.run(checks.run(args))
    report = json.loads((tmp_path / "performance.json").read_text())
    assert report["passed"]
    assert report["stages"][0]["completed"] == 4
    assert report["stages"][0]["output_tokens"] == 4 * 256
