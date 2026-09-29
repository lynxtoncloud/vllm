# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in eager input-preparation fences, without GPU work during setup."""

import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_active_trace: ContextVar["InputPreparationTrace | None"] = ContextVar(
    "active_input_preparation_trace", default=None
)


def gpu_memory_snapshot(device: torch.device) -> dict[str, int]:
    free, total = torch.accelerator.get_memory_info(device)
    return {
        "free_bytes": free,
        "total_bytes": total,
        "allocated_bytes": torch.accelerator.memory_allocated(device),
        "reserved_bytes": torch.accelerator.memory_reserved(device),
    }


def is_input_trace_active() -> bool:
    return _active_trace.get() is not None


@contextmanager
def activate_input_trace(trace: "InputPreparationTrace | None"):
    token = _active_trace.set(trace)
    try:
        yield
    finally:
        _active_trace.reset(token)


@contextmanager
def trace_span(stage: str, **details: object):
    trace = _active_trace.get()
    if trace is None:
        yield
    else:
        with trace.span(stage, **details):
            yield


class InputPreparationTrace:
    """Locate asynchronous faults using injected device synchronization.

    Details must contain JSON-serializable CPU values. Never stringify tensors:
    doing so could trigger a GPU read before the checkpoint being diagnosed.
    """

    def __init__(
        self,
        rank: int,
        enforce_eager: bool,
        synchronize: Callable[[], None],
        memory_snapshot: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        if not enforce_eager:
            raise ValueError("Input preparation tracing requires eager execution")
        for name in (
            "AMD_SERIALIZE_KERNEL",
            "AMD_SERIALIZE_COPY",
            "HIP_LAUNCH_BLOCKING",
            "CUDA_LAUNCH_BLOCKING",
        ):
            if os.environ.get(name, "0").strip() not in ("", "0"):
                raise ValueError(
                    f"Input preparation tracing requires {name} unset or 0; "
                    "global launch blocking can stall collective initialization"
                )
        self.rank = rank
        self.ready = False
        self.step = 0
        self.synchronize = synchronize
        self.memory_snapshot = memory_snapshot
        self.request_ids: list[str] = []

    def _emit(self, stage: str, event: str, **cpu_details: object) -> None:
        row = {
            "rank": self.rank,
            "pid": os.getpid(),
            "step": self.step,
            "time": time.time(),
            "stage": stage,
            "event": event,
            "details": cpu_details,
        }
        print(
            "Input preparation trace: "
            + json.dumps(row, ensure_ascii=False, allow_nan=False),
            flush=True,
        )

    def begin(self, scheduled_tokens: dict[str, int]) -> None:
        self.step += 1
        self.request_ids = list(scheduled_tokens)
        self._emit(
            "batch",
            "BEGIN",
            requests=len(scheduled_tokens),
            tokens=sum(scheduled_tokens.values()),
            scheduled_tokens=scheduled_tokens,
        )
        self.checkpoint("entry")

    def checkpoint(self, stage: str, **cpu_details: object) -> None:
        self._emit(stage, "BEFORE", **cpu_details)
        try:
            self.synchronize()
        except Exception as error:
            self._emit(stage, "FAILED", error=f"{type(error).__name__}: {error}")
            raise
        self._emit(stage, "DONE", **cpu_details)

    @contextmanager
    def span(self, stage: str, **details: object):
        details = {"request_ids": self.request_ids, **details}
        self._emit(stage, "BEGIN", **details)
        start = time.perf_counter()
        try:
            self.synchronize()
            memory_before = self._memory_snapshot()
            ready_ms = round((time.perf_counter() - start) * 1000, 3)
            self._emit(
                stage,
                "READY",
                ready_ms=ready_ms,
                memory_before=memory_before,
                **details,
            )
            body_start = time.perf_counter()
            yield
            body_ms = round((time.perf_counter() - body_start) * 1000, 3)
            self._emit(stage, "EXECUTED", body_ms=body_ms, **details)
            sync_start = time.perf_counter()
            self.synchronize()
            sync_ms = round((time.perf_counter() - sync_start) * 1000, 3)
            memory_after = self._memory_snapshot()
        except Exception as error:
            self._emit(
                stage,
                "FAILED",
                elapsed_ms=round((time.perf_counter() - start) * 1000, 3),
                error=f"{type(error).__name__}: {error}",
                **details,
            )
            raise
        self._emit(
            stage,
            "DONE",
            elapsed_ms=round((time.perf_counter() - start) * 1000, 3),
            ready_ms=ready_ms,
            body_ms=body_ms,
            sync_ms=sync_ms,
            memory_before=memory_before,
            memory_after=memory_after,
            **details,
        )

    def _memory_snapshot(self) -> dict[str, int] | dict[str, str] | None:
        if self.memory_snapshot is None:
            return None
        try:
            return self.memory_snapshot()
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}
