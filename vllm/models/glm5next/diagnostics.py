# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in eager diagnostics for the first non-finite GLM activation."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def install_finite_checks(
    model: nn.Module,
    *,
    rank: int,
    enforce_eager: bool,
    active: Callable[[], bool],
    module_types: tuple[type[nn.Module], ...],
) -> int:
    """Check selected module boundaries and their direct parameters on first use.

    Only run when ``active`` identifies a real forward pass. Hooks synchronize
    the device and intentionally fail the diagnostic engine on non-finite data.
    Buffers are not scanned: unused KV slots may contain arbitrary values.
    """
    if not enforce_eager:
        raise ValueError("VLLM_GLM5NEXT_CHECK_FINITE requires --enforce-eager")

    def check(value: Any, location: str) -> None:
        if isinstance(value, torch.Tensor):
            if not value.is_floating_point() or value.numel() == 0:
                return
            finite = torch.isfinite(value)
            if not finite.all().item():
                count = (~finite).sum().item()
                raise RuntimeError(
                    f"GLM non-finite: tp_rank={rank} {location} "
                    f"shape={tuple(value.shape)} dtype={value.dtype} "
                    f"device={value.device} bad={count}/{value.numel()}"
                )
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                check(item, f"{location}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                check(item, f"{location}[{key}]")

    def attach(module: nn.Module, name: str) -> None:
        checked_parameters = False

        def before(module, args, kwargs):
            nonlocal checked_parameters
            if not active():
                return
            check(args, f"module={name} input")
            check(kwargs, f"module={name} kwargs")
            if not checked_parameters:
                for parameter_name, parameter in module.named_parameters(recurse=False):
                    check(parameter, f"module={name} parameter={parameter_name}")
                checked_parameters = True

        def after(module, args, output):
            if active():
                check(output, f"module={name} output")

        module.register_forward_pre_hook(before, with_kwargs=True)
        module.register_forward_hook(after)

    count = 0
    for name, module in model.named_modules():
        if (
            not name
            or isinstance(module, module_types)
            or name.rsplit(".", 1)[-1] in ("self_attn", "mlp")
        ):
            attach(module, name or "model")
            count += 1
    return count
