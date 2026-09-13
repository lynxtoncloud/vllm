# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in HTTP acceptance tests; no GPU imports, remote restarts or fault injection.

See examples/disaggregated/glm53_gfx1100/ACCEPTANCE.md. Without
GLM53_PD_TEST_URL only response-parser tests run. Do not run with pytest-xdist:
metric deltas and cache-isolation cases require an otherwise idle service.
"""

import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
import regex as re

MODEL = "zai-org/GLM-5.3-Flash-BF16"
METRICS = (
    "vllm:nixl_bytes_transferred_count",
    "vllm:nixl_bytes_transferred_sum",
    "vllm:nixl_num_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total",
)


def metric_values(text, model):
    result: dict[str, float] = {}
    for line in text.splitlines():
        match = re.match(r"^(vllm:nixl_\w+)\{([^}]*)\}\s+(\S+)", line)
        if not match or match[1] not in METRICS:
            continue
        labels = dict(re.findall(r'(\w+)="([^"\\]*)"', match[2]))
        if labels.get("model_name") != model:
            continue
        value = float(match[3])
        assert math.isfinite(value) and value >= 0, line
        result[match[1]] = result.get(match[1], 0) + value
    assert set(result) == set(METRICS), f"Missing NIXL metrics: {result}"
    return result


def sse_payloads(lines):
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


class Service:
    def __init__(self, url, results):
        self.url = url.rstrip("/")
        self.results = results
        self.model = os.getenv("GLM53_TEST_MODEL", MODEL)
        # Deliberately bypass shell HTTP proxies for internal service traffic.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.timeout = float(os.getenv("GLM53_TEST_TIMEOUT", "300"))

    def record(self, record):
        with self.results.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def request(self, path, body=None, *, stream=False):
        headers = {"Content-Type": "application/json"}
        if key := os.getenv("GLM53_TEST_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
        )
        started = time.perf_counter()
        raw: str | list[str]
        content_type = ""
        try:
            with self.http.open(req, timeout=self.timeout) as response:
                status = response.status
                content_type = response.headers.get("Content-Type", "")
                if stream:
                    raw = list(sse_payloads(response))
                else:
                    raw = response.read().decode()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read().decode()
        except Exception as exc:
            self.record(
                dict(
                    at=time.time(),
                    url=self.url,
                    path=path,
                    request=body,
                    elapsed_s=time.perf_counter() - started,
                    transport_error=repr(exc),
                )
            )
            raise
        self.record(
            dict(
                at=time.time(),
                url=self.url,
                path=path,
                status=status,
                elapsed_s=time.perf_counter() - started,
                request=body,
                response=raw,
                content_type=content_type,
            )
        )
        if stream and status == 200:
            assert "text/event-stream" in content_type, content_type
        return status, raw

    def json(self, path, body=None):
        status, raw = self.request(path, body)
        assert status == 200, (status, raw)
        result = json.loads(raw)
        assert "error" not in result, result
        return result

    def complete(self, prompt, **kwargs):
        body = dict(
            model=self.model, prompt=prompt, temperature=0, max_tokens=16, stream=False
        )
        body.update(kwargs)
        result = self.json("/v1/completions", body)
        assert result["model"] == self.model
        choice = result["choices"][0]
        assert choice["finish_reason"] in ("stop", "length"), choice
        assert isinstance(choice["text"], str)
        usage = result["usage"]
        assert 0 < usage["completion_tokens"] <= body["max_tokens"]
        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )
        return result

    def chat(self, messages, **kwargs):
        body = dict(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=128,
            stream=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        body.update(kwargs)
        return self.json("/v1/chat/completions", body)


@pytest.fixture(scope="module")
def services(tmp_path_factory):
    url = os.getenv("GLM53_PD_TEST_URL")
    if not url:
        pytest.skip("Set GLM53_PD_TEST_URL explicitly to exercise live services")
    if os.getenv("PYTEST_XDIST_WORKER"):
        pytest.fail("Run acceptance tests without xdist")
    results = Path(
        os.getenv(
            "GLM53_TEST_RESULTS",
            str(tmp_path_factory.mktemp("glm53") / "requests.jsonl"),
        )
    )
    results.parent.mkdir(parents=True, exist_ok=True)
    urls = dict(
        proxy=url,
        prefill=os.getenv("GLM53_PREFILL_URL", "http://10.5.10.36:8001"),
        decode=os.getenv("GLM53_DECODE_URL", "http://10.5.10.55:8002"),
    )
    return {name: Service(address, results) for name, address in urls.items()}


class TestHealth:
    def test_engine_health_and_model_identity(self, services):
        for name in ("prefill", "decode"):
            service = services[name]
            assert service.request("/health")[0] == 200
            models = service.json("/v1/models")["data"]
            assert service.model in [m["id"] for m in models]


class TestFunctional:
    @pytest.mark.parametrize(
        "prompt,answer",
        [
            ("The capital of France is", "Paris"),
            ("问题：中国的首都是哪里？\n简短答案：", "北京"),
            ("Calculate 17 + 25. Answer:", "42"),
            ("资料：订单 A17 的状态为已发货。\n问题：A17 的状态？\n答案：", "已发货"),
        ],
    )
    def test_completion_semantics(self, services, prompt, answer):
        result = services["proxy"].complete(prompt, max_tokens=32)
        assert answer in result["choices"][0]["text"], result

    def test_multiturn_correction(self, services):
        result = services["proxy"].chat(
            [
                {"role": "system", "content": "只输出用户最后确认的订单编号。"},
                {"role": "user", "content": "订单编号是 AX731。"},
                {"role": "assistant", "content": "已记录 AX731。"},
                {"role": "user", "content": "更正为 BY942。最终编号是什么？"},
            ]
        )
        text = result["choices"][0]["message"]["content"]
        assert "BY942" in text and "AX731" not in text, result

    def test_json_schema(self, services):
        result = services["proxy"].chat(
            [{"role": "user", "content": "订单编号 A17，数量 3。提取为 JSON。"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "order",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "quantity": {"type": "integer"},
                        },
                        "required": ["id", "quantity"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        assert json.loads(result["choices"][0]["message"]["content"]) == {
            "id": "A17",
            "quantity": 3,
        }

    def test_tool_call_and_result(self, services):
        messages = [{"role": "user", "content": "Call lookup_order for A17."}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Look up an order by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            }
        ]
        result = services["proxy"].chat(messages, tools=tools, tool_choice="required")
        message = result["choices"][0]["message"]
        assert result["choices"][0]["finish_reason"] == "tool_calls", result
        calls = message["tool_calls"]
        assert len(calls) == 1 and calls[0]["function"]["name"] == "lookup_order"
        assert json.loads(calls[0]["function"]["arguments"]) == {"order_id": "A17"}
        messages += [
            {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": calls,
            },
            {
                "role": "tool",
                "tool_call_id": calls[0]["id"],
                "content": '{"status":"shipped"}',
            },
        ]
        final = services["proxy"].chat(messages, tools=tools, tool_choice="none")
        assert "shipped" in final["choices"][0]["message"]["content"].lower()

    def test_sequential_request_isolation(self, services):
        prefix = "Record: invoice_id=ZX173; marker="
        for marker in ("ALPHA731", "BETA942", "ALPHA731"):
            result = services["proxy"].complete(
                prefix + marker + ".\nRepeat only the marker:\n", max_tokens=16
            )
            text = result["choices"][0]["text"]
            assert marker in text, result
            other = "BETA942" if marker == "ALPHA731" else "ALPHA731"
            assert other not in text, result


class TestTransfer:
    def test_fresh_request_increases_nixl_bytes_without_failures(self, services):
        decoder = services["decode"]
        before = metric_values(decoder.request("/metrics")[1], decoder.model)
        services["proxy"].complete(
            f"Trace {uuid.uuid4().hex}. The capital of France is", max_tokens=16
        )
        deadline = time.monotonic() + 30
        while True:
            after = metric_values(decoder.request("/metrics")[1], decoder.model)
            assert all(after[k] >= before[k] for k in METRICS), "Counters reset"
            assert all(after[k] == before[k] for k in METRICS[2:]), (before, after)
            if all(after[k] > before[k] for k in METRICS[:2]):
                break
            assert time.monotonic() < deadline, (before, after)
            time.sleep(1)
        decoder.record(dict(case="PD-01", before=before, after=after))


class TestBoundary:
    @pytest.mark.parametrize(
        "length",
        [
            1,
            3,
            4,
            5,
            6,
            7,
            127,
            128,
            129,
            511,
            512,
            513,
            514,
            515,
            1023,
            1024,
            1025,
            2044,
            2045,
            2046,
            2047,
        ],
    )
    def test_exact_token_lengths(self, services, length):
        p = services["prefill"]
        ids = p.json(
            "/tokenize", dict(model=p.model, prompt=" hello", add_special_tokens=False)
        )["tokens"]
        assert ids
        prompt = (ids * length)[:length]
        result = services["proxy"].complete(prompt, max_tokens=min(8, 2048 - length))
        assert result["usage"]["prompt_tokens"] == length, result


class TestParity:
    @pytest.mark.parametrize("length", [3, 4, 5, 511, 512, 513, 2047])
    def test_pd_matches_d_local_prefill(self, services, length):
        """D without kv_transfer_params is the upstream local-prefill baseline."""
        p = services["prefill"]
        ids = p.json(
            "/tokenize", dict(model=p.model, prompt=" hello", add_special_tokens=False)
        )["tokens"]
        assert ids
        prompt = (ids * length)[:length]
        options = dict(max_tokens=min(8, 2048 - length), logprobs=1, seed=42)
        pd = services["proxy"].complete(prompt, **options)["choices"][0]
        local = services["decode"].complete(prompt, **options)["choices"][0]
        assert pd["text"] == local["text"], (pd, local)
        assert pd["finish_reason"] == local["finish_reason"]
        assert pd["logprobs"]["tokens"] == local["logprobs"]["tokens"]
        a, b = pd["logprobs"]["token_logprobs"], local["logprobs"]["token_logprobs"]
        assert a and len(a) == len(b)
        tolerance = float(os.getenv("GLM53_LOGPROB_ATOL", "0.02"))
        assert all(
            math.isfinite(x) and math.isfinite(y) and abs(x - y) <= tolerance
            for x, y in zip(a, b)
        ), (a, b)


class TestGateway:
    """Public API contracts: the current toy proxy is expected to fail some."""

    @pytest.mark.parametrize(
        "change,status",
        [
            ({"model": "nonexistent-model"}, 404),
            ({"max_tokens": -1}, 400),
            ({"max_tokens": 4096}, 400),
            ({"temperature": -1}, 400),
        ],
    )
    def test_validation_errors_are_client_errors(self, services, change, status):
        body = dict(model=services["proxy"].model, prompt="hello", max_tokens=8)
        body.update(change)
        actual, raw = services["proxy"].request("/v1/completions", body)
        assert actual == status, (actual, raw)
        assert "error" in json.loads(raw), raw

    def test_context_overflow_is_rejected(self, services):
        p = services["prefill"]
        ids = p.json(
            "/tokenize", dict(model=p.model, prompt=" hello", add_special_tokens=False)
        )["tokens"]
        assert ids
        body = dict(model=p.model, prompt=(ids * 2049)[:2049], max_tokens=1)
        status, raw = services["proxy"].request("/v1/completions", body)
        assert status == 400, (status, raw)

    def test_sse_content_type_termination_and_text(self, services):
        body = dict(
            model=services["proxy"].model,
            prompt="The capital of France is",
            max_tokens=16,
            temperature=0,
            stream=True,
        )
        status, events = services["proxy"].request("/v1/completions", body, stream=True)
        assert status == 200 and events[-1] == "[DONE]", events
        chunks = [json.loads(e) for e in events[:-1]]
        assert all("error" not in c for c in chunks)
        choices = [c for chunk in chunks for c in chunk.get("choices", [])]
        assert "Paris" in "".join(c.get("text", "") for c in choices)
        assert any(c.get("finish_reason") in ("stop", "length") for c in choices)


def test_metric_parser_aggregates_only_target_model():
    lines = [
        f'{key}{{engine="{engine}",model_name="{model}"}} {value}'
        for key in METRICS
        for engine, model, value in (
            ("0", MODEL, 2),
            ("1", MODEL, 3),
            ("0", "other", 999),
        )
    ]
    assert metric_values("\n".join(lines), MODEL) == dict.fromkeys(METRICS, 5)
    with pytest.raises(AssertionError, match="Missing NIXL"):
        metric_values("", MODEL)


def test_sse_parser_handles_comments_crlf_and_multiline_json():
    lines = [
        b": ping\r\n",
        b'data: {"a":\r\n',
        b"data: 1}\r\n",
        b"\r\n",
        b"data: [DONE]\r\n",
        b"\r\n",
    ]
    assert list(sse_payloads(lines)) == ['{"a":\n1}', "[DONE]"]


def test_http_client_preserves_upstream_error_body(tmp_path, monkeypatch):
    from email.message import Message
    from io import BytesIO

    service = Service("http://test.invalid", tmp_path / "requests.jsonl")

    def reject(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://test.invalid",
            400,
            "Bad Request",
            Message(),
            BytesIO(b'{"error":{"message":"context overflow"}}'),
        )

    monkeypatch.setattr(service.http, "open", reject)
    status, raw = service.request("/v1/completions", {"max_tokens": 4096})
    assert status == 400 and json.loads(raw)["error"]["message"] == "context overflow"
    record = json.loads(service.results.read_text())
    assert record["status"] == 400 and record["request"]["max_tokens"] == 4096


def test_http_client_records_incomplete_transfer(tmp_path, monkeypatch):
    service = Service("http://test.invalid", tmp_path / "requests.jsonl")

    def disconnect(*args, **kwargs):
        raise ConnectionResetError("synthetic upstream disconnect")

    monkeypatch.setattr(service.http, "open", disconnect)
    with pytest.raises(ConnectionResetError):
        service.request("/v1/completions", {"prompt": "synthetic"})
    assert (
        "ConnectionResetError"
        in json.loads(service.results.read_text())["transport_error"]
    )
