# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.v1.worker.gpu.input_trace import (
    InputPreparationTrace,
    activate_input_trace,
    trace_span,
)

pytestmark = pytest.mark.cpu_test


def test_trace_span_records_synchronized_memory_and_request(capsys):
    syncs: list[None] = []
    snapshots = iter([{"free_bytes": 100}, {"free_bytes": 80}])
    trace = InputPreparationTrace(
        rank=2,
        enforce_eager=True,
        synchronize=lambda: syncs.append(None),
        memory_snapshot=lambda: next(snapshots),
    )
    trace.begin({"request-1": 4})

    with activate_input_trace(trace), trace_span("vision_block", block=0):
        pass

    rows = [
        json.loads(line.removeprefix("Input preparation trace: "))
        for line in capsys.readouterr().out.splitlines()
    ]
    span_rows = [row for row in rows if row["stage"] == "vision_block"]
    assert [row["event"] for row in span_rows] == [
        "BEGIN",
        "READY",
        "EXECUTED",
        "DONE",
    ]
    assert span_rows[-1]["details"]["request_ids"] == ["request-1"]
    assert span_rows[-1]["details"]["memory_before"] == {"free_bytes": 100}
    assert span_rows[-1]["details"]["memory_after"] == {"free_bytes": 80}
    assert span_rows[-1]["details"]["elapsed_ms"] >= 0
    assert len(syncs) == 3


def test_trace_span_reports_failure_and_clears_context(capsys):
    trace = InputPreparationTrace(rank=0, enforce_eager=True, synchronize=lambda: None)
    trace.begin({"request-2": 1})

    with (
        pytest.raises(ValueError, match="failed"),
        activate_input_trace(trace),
        trace_span("vision_patch_embed"),
    ):
        raise ValueError("failed")

    with trace_span("inactive"):
        pass

    rows = [
        json.loads(line.removeprefix("Input preparation trace: "))
        for line in capsys.readouterr().out.splitlines()
    ]
    assert rows[-1]["stage"] == "vision_patch_embed"
    assert rows[-1]["event"] == "FAILED"
    assert rows[-1]["details"]["error"] == "ValueError: failed"


def test_trace_span_keeps_running_when_memory_snapshot_fails(capsys):
    def unavailable():
        raise RuntimeError("memory stats unavailable")

    trace = InputPreparationTrace(
        rank=0,
        enforce_eager=True,
        synchronize=lambda: None,
        memory_snapshot=unavailable,
    )
    trace.begin({"request-3": 1})

    with activate_input_trace(trace), trace_span("vision_merger"):
        pass

    rows = [
        json.loads(line.removeprefix("Input preparation trace: "))
        for line in capsys.readouterr().out.splitlines()
    ]
    assert rows[-1]["event"] == "DONE"
    assert rows[-1]["details"]["memory_before"] == {
        "error": "RuntimeError: memory stats unavailable"
    }
