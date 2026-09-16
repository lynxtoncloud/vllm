# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in eager input-preparation fences, without GPU work during setup."""

import json
import os
import time
from collections.abc import Callable


class InputPreparationTrace:
    """Locate asynchronous faults using injected device synchronization.

    Details must contain JSON-serializable CPU values. Never stringify tensors:
    doing so could trigger a GPU read before the checkpoint being diagnosed.
    """

    def __init__(
        self, rank: int, enforce_eager: bool, synchronize: Callable[[], None]
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
