# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in eager diagnostics for the first non-finite GLM activation."""

from collections.abc import Callable
from functools import wraps
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
                # RoutedExperts methods bypass nn.Module.__call__, so their
                # parameter checks must run at the owning MoERunner boundary.
                routed = getattr(module, "routed_experts", None)
                if isinstance(routed, nn.Module):
                    for parameter_name, parameter in routed.named_parameters():
                        check(
                            parameter,
                            f"module={name}.routed_experts parameter={parameter_name}",
                        )
                    kernel = getattr(
                        getattr(routed, "quant_method", None), "moe_kernel", None
                    )
                    experts = getattr(kernel, "fused_experts", None)
                    if type(experts).__name__ == "TritonWNA16Experts":
                        trace_wna16_stages(experts, name)
                checked_parameters = True

        def after(module, args, output):
            if active():
                check(output, f"module={name} output")

        module.register_forward_pre_hook(before, with_kwargs=True)
        module.register_forward_hook(after)

    def trace_method(owner: Any, method_name: str, name: str) -> None:
        method = getattr(owner, method_name)

        @wraps(method)
        def traced(*args, **kwargs):
            enabled = active()
            location = f"module={name} stage={method_name}"
            if enabled:
                check(args, f"{location} input")
                check(kwargs, f"{location} kwargs")
            output = method(*args, **kwargs)
            if enabled:
                check(output, f"{location} output")
            return output

        setattr(owner, method_name, traced)

    def trace_wna16_stages(experts: Any, name: str) -> None:
        activation_fn = experts.activation
        sum_fn = experts.moe_sum

        @wraps(activation_fn)
        def activation(activation, output, input, **kwargs):
            enabled = active()
            if enabled:
                check(input, f"module={name} stage=gemm1 output")
            activation_fn(activation, output, input, **kwargs)
            if enabled:
                check(output, f"module={name} stage=activation output")

        @wraps(sum_fn)
        def moe_sum(input, output):
            enabled = active()
            if enabled:
                check(input, f"module={name} stage=gemm2 output")
            sum_fn(input, output)
            if enabled:
                check(output, f"module={name} stage=moe_sum output")

        experts.activation = activation
        experts.moe_sum = moe_sum

    count = 0
    for name, module in model.named_modules():
        is_moe_runner = hasattr(module, "_apply_quant_method")
        if (
            not name
            or is_moe_runner
            or isinstance(module, module_types)
            or name.rsplit(".", 1)[-1] in ("self_attn", "mlp", "gate")
        ):
            attach(module, name or "model")
            count += 1
        if is_moe_runner:
            for method_name in (
                "_apply_quant_method",
                "_maybe_reduce_routed_output_before_transform",
                "_maybe_reduce_shared_expert_output",
                "_maybe_apply_routed_scale_to_output",
                "_maybe_reduce_final_output",
            ):
                if hasattr(module, method_name):
                    trace_method(module, method_name, name)
            trace_method(module.router, "select_experts", f"{name}.router")
            trace_method(
                module.routed_experts, "forward_modular", f"{name}.routed_experts"
            )
    return count
