# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm grouped W8A16 GEMM consuming CT offset-binary packed weights."""

import functools
import json
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.model_executor.parameter import permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


def validate_launch_config(config):
    allowed = {
        "BLOCK_M": (16, 32, 64),
        "BLOCK_N": (32, 64, 128),
        "BLOCK_K": (32, 64, 128),
        "num_warps": (4, 8),
    }
    if set(config) != set(allowed) or any(
        type(config[key]) is not int or config[key] not in values
        for key, values in allowed.items()
    ):
        raise ValueError(f"Invalid W8A16 launch config: {config}")
    return config


@functools.lru_cache
def load_tuned_configs(path, device_name, arch):
    data = json.loads(Path(path).read_text())
    if data.get("device") != device_name or data.get("arch") != arch:
        raise ValueError("W8A16 tuning file GPU does not match this worker")
    if data.get("version") != 1:
        raise ValueError("Unsupported W8A16 tuning file version")
    return {
        key: validate_launch_config(value) for key, value in data["configs"].items()
    }


def launch_config(m, n, k, dtype, device):
    default = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=32, num_warps=4)
    if not envs.VLLM_ROCM_W8A16_CONFIG:
        return default
    properties = torch.cuda.get_device_properties(device)
    configs = load_tuned_configs(
        envs.VLLM_ROCM_W8A16_CONFIG, properties.name, properties.gcnArchName
    )
    # Only use measured shapes; interpolation can regress decode or graph padding.
    return configs.get(f"{dtype}:{m}:{n}:{k}", default)


@triton.jit
def _w8a16_gemm(
    X,
    W,
    S,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start * BLOCK_K + rk
        x = tl.load(
            X + m[:, None] * stride_xm + k[None, :] * stride_xk,
            (m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        packed = tl.load(
            W + n[None, :] * (K // 4) + k[:, None] // 4,
            (n[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        q = ((packed >> ((k[:, None] % 4) * 8)) & 255) - 128
        scale = tl.load(
            S + n[None, :] * (K // GROUP_SIZE) + k[:, None] // GROUP_SIZE,
            (n[None, :] < N) & (k[:, None] < K),
            other=0,
        ).to(tl.float32)
        w = (q.to(tl.float32) * scale).to(x.dtype)
        acc = tl.dot(x, w, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


class TritonW8A16LinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "W8A16 Triton requires ROCm"
        if c.weight_type != scalar_types.uint8b128 or c.zero_points:
            return False, "Only symmetric offset-binary INT8 weights are supported"
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "W8A16 requires FP16 or BF16 activations"
        if c.out_type not in (None, c.act_type):
            return False, "Output and activation dtypes must match"
        k, n = c.partition_weight_shape
        if c.group_size != 128 or k <= 0 or n <= 0 or k % 128:
            return False, "W8A16 requires complete 128-element groups on each TP rank"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        def canonical_weight(param):
            permute_param_layout_(param, input_dim=1, output_dim=0, packed_dim=1)
            return param.contiguous()

        def canonical_scale(param):
            permute_param_layout_(param, input_dim=1, output_dim=0)
            return param.contiguous()

        self._transform_param(layer, self.w_q_name, canonical_weight)
        self._transform_param(layer, self.w_s_name, canonical_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k, n = self.config.partition_weight_shape
        w, scales, _ = self._get_weight_params(layer)
        assert x.shape[-1] == k and x.dtype == self.config.act_type
        assert w.dtype == torch.int32 and w.shape == (n, k // 4)
        assert scales.shape == (n, k // 128)
        x2d = x.reshape(-1, k)
        output = torch.empty((x2d.shape[0], n), dtype=x.dtype, device=x.device)
        if x2d.shape[0]:
            config = launch_config(x2d.shape[0], n, k, x.dtype, x.device)
            _w8a16_gemm[
                (
                    triton.cdiv(x2d.shape[0], config["BLOCK_M"]),
                    triton.cdiv(n, config["BLOCK_N"]),
                )
            ](
                x2d,
                w,
                scales,
                output,
                x2d.shape[0],
                n,
                k,
                x2d.stride(0),
                x2d.stride(1),
                GROUP_SIZE=128,
                **config,
            )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*x.shape[:-1], n)
