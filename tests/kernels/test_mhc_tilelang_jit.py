# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vllm.platforms import current_platform
from vllm.utils import import_utils


class _PassConfigKey:
    TL_DISABLE_WARP_SPECIALIZED = "disable_warp_specialized"
    TL_DISABLE_TMA_LOWER = "disable_tma_lower"
    TL_PTXAS_REGISTER_USAGE_LEVEL = "ptxas_register_usage_level"


def _install_tilelang_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    calls = {"jit_decorate": 0, "compiled_call": 0, "compile": 0, "cache_miss": 0}

    tilelang: Any = ModuleType("tilelang")

    def jit(**kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            calls["jit_decorate"] += 1
            cache: dict[Any, Any] = {}

            def parse_args(*args: Any, **kw: Any) -> Any:
                return (args, tuple(sorted(kw.items()))), None

            def compile_kernel(*args: Any, **kw: Any) -> Any:
                calls["compile"] += 1
                return object()

            def compiled(*args: Any, **kw: Any) -> Any:
                calls["compiled_call"] += 1
                key, _ = parse_args(*args, **kw)
                if key not in cache:
                    calls["cache_miss"] += 1
                    cache[key] = compile_kernel(*args, **kw)
                return func.__name__

            jit_impl: Any = compiled
            jit_impl.compile = compile_kernel
            jit_impl.func = SimpleNamespace(parse_args=parse_args)
            jit_impl._kernel_cache = cache
            return jit_impl

        return decorate

    tilelang.PassConfigKey = _PassConfigKey
    tilelang.jit = jit

    monkeypatch.setattr(import_utils, "has_tilelang", lambda: True)
    monkeypatch.setitem(sys.modules, "tilelang", tilelang)
    monkeypatch.setitem(
        sys.modules, "tilelang.language", ModuleType("tilelang.language")
    )
    monkeypatch.delitem(sys.modules, "vllm.tilelang_utils", raising=False)

    return calls


def test_tilelang_jit_decorator_is_lazy_only_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (current_platform.is_cuda() or current_platform.is_rocm()):
        pytest.skip("Test requires CUDA or ROCm")

    calls = _install_tilelang_stub(monkeypatch)
    module_name = "vllm.model_executor.kernels.mhc.tilelang_kernels"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)

    if current_platform.is_rocm():
        assert calls["jit_decorate"] == 0
    else:
        assert calls["jit_decorate"] > 0

    decorated_calls = calls["jit_decorate"]
    assert module.mhc_post_tilelang() == "mhc_post_tilelang"
    if current_platform.is_rocm():
        assert calls["jit_decorate"] == 1
    else:
        assert calls["jit_decorate"] == decorated_calls
    assert calls["compiled_call"] == 1


@pytest.mark.parametrize("is_rocm", [True, False])
@pytest.mark.parametrize("runtime_first", [True, False])
def test_tilelang_compile_only_warmup_populates_runtime_cache(
    monkeypatch: pytest.MonkeyPatch, is_rocm: bool, runtime_first: bool
) -> None:
    """Warmup must compile without launching, including before the first call."""
    from vllm.model_executor.warmup.jit_warmup_tilelang_helper import compile_tilelang

    monkeypatch.setattr(current_platform, "is_rocm", lambda: is_rocm)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: not is_rocm)
    calls = _install_tilelang_stub(monkeypatch)
    module = importlib.import_module("vllm.tilelang_utils")

    def sample_kernel(value: int, *, scale: int = 1) -> None:
        pass

    # TileLang binds language globals in the decorated function's module.
    monkeypatch.setitem(sample_kernel.__globals__, "T", None)
    monkeypatch.setitem(sample_kernel.__globals__, "tilelang", None)
    kernel = module.tilelang_jit(sample_kernel)
    assert calls["jit_decorate"] == (0 if is_rocm else 1)

    if runtime_first:
        kernel(1, scale=2)
    launches = calls["compiled_call"]
    misses = calls["cache_miss"]

    compile_tilelang(kernel, 3, scale=4)

    assert calls["compiled_call"] == launches
    assert calls["compile"] == int(runtime_first) + 1
    if is_rocm:
        assert sample_kernel.__globals__["T"] is module.T
    assert kernel(3, scale=4) == "sample_kernel"
    assert kernel(3, scale=4) == "sample_kernel"
    assert calls["cache_miss"] == misses
    assert calls["jit_decorate"] == 1


@pytest.mark.skipif(not current_platform.is_rocm(), reason="Test requires ROCm")
def test_deepseek_v4_import_and_jit_monitor_do_not_hijack_hip_symbols() -> None:
    if importlib.util.find_spec("tilelang") is None:
        pytest.skip("Test requires TileLang to be installed")

    # Both claims are about process-global state, `sys.modules` and the symbol
    # table, and a sibling test legitimately imports TileLang to exercise those
    # kernels, so the checks only mean something in an interpreter of their own.
    script = Path(__file__).parent / "scripts" / "check_no_tilelang_hijack.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"HIP symbols were hijacked:\n{result.stdout}\n{result.stderr}")
