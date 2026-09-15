# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CT W8A16 load-to-GEMM regressions, including TP shards and partial tiles."""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("Requires a ROCm GPU", allow_module_level=True)

from vllm.model_executor import parameter as parameter_mod
from vllm.model_executor.kernels.linear.mixed_precision.triton_w8a16 import (
    TritonW8A16LinearKernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
    CompressedTensorsWNA16,
)


@pytest.mark.parametrize("m", [0, 1, 17, 33])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("parallel", ["replicated", "row", "column", "merged_column"])
def test_ct_w8a16_tp_loading_and_gemm(monkeypatch, m, dtype, parallel):
    """Use the CT scheme and real parameter loaders before calling the kernel."""
    torch.manual_seed(17)
    merged = parallel == "merged_column"
    column = parallel in ("column", "merged_column")
    k, n = 512, 140 if merged else 70
    q = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int32)
    packed = torch.zeros((n, k // 4), device="cuda", dtype=torch.int32)
    for i in range(4):
        packed |= (q[:, i::4] + 128) << (8 * i)
    scales = (torch.rand(n, k // 128, device="cuda") * 0.01 + 0.001).to(dtype)
    weight = (q.float() * scales.float().repeat_interleave(128, 1)).to(dtype)
    # Noncontiguous input plus a leading batch dimension exercises reshape/strides.
    x = torch.randn((1, m, k * 2), device="cuda", dtype=dtype)[..., ::2] / 10
    bias = torch.randn(n, device="cuda", dtype=dtype) / 10
    size = 1 if parallel == "replicated" else 2
    outputs = []
    for rank in range(size):
        monkeypatch.setattr(
            parameter_mod, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        monkeypatch.setattr(
            parameter_mod, "get_tensor_model_parallel_world_size", lambda: size
        )
        local_k = k // size if parallel == "row" else k
        local_n = n // size if column else n
        scheme = CompressedTensorsWNA16("group", 8, 128, True)
        layer = torch.nn.Module()
        with torch.device("cuda"):
            scheme.create_weights(
                layer,
                n,
                k,
                [local_n // 2] * 2 if merged else [local_n],
                local_k,
                dtype,
                weight_loader=lambda param, value: param.data.copy_(value),
            )
        assert isinstance(scheme.kernel, TritonW8A16LinearKernel)
        for param, value in [
            (layer.weight_packed, packed),
            (layer.weight_scale, scales),
        ]:
            if merged:
                # The PR loads gate/up independently into a fused parameter.
                # Each shard must be sliced by TP before concatenation.
                for shard, value_shard in enumerate(value.chunk(2, dim=0)):
                    param.load_merged_column_weight(
                        value_shard,
                        shard_offset=shard * (local_n // 2),
                        shard_size=local_n // 2,
                    )
            elif parallel == "row":
                param.load_row_parallel_weight(value)
            else:
                param.load_column_parallel_weight(value)
        scheme.process_weights_after_loading(layer)
        assert layer.weight_packed.dtype == torch.int32
        assert layer.weight_packed.numel() == local_n * local_k // 4
        local_x = (
            x[..., rank * local_k : (rank + 1) * local_k] if parallel == "row" else x
        )
        if merged:
            local_bias = bias.view(2, -1)[
                :, rank * (local_n // 2) : (rank + 1) * (local_n // 2)
            ].reshape(-1)
        else:
            local_bias = bias[rank * local_n : (rank + 1) * local_n] if column else bias
        out = scheme.apply_weights(
            layer, local_x, None if parallel == "row" else local_bias
        )
        outputs.append(out.float())
    if merged:
        actual = torch.cat(
            [
                torch.cat([out.chunk(2, dim=-1)[shard] for out in outputs], dim=-1)
                for shard in range(2)
            ],
            dim=-1,
        )
    else:
        actual = (
            (sum(outputs) + bias.float())
            if parallel == "row"
            else torch.cat(outputs, dim=-1)
        )
    expected = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("m", [1, 17, 32])
@pytest.mark.parametrize(
    "block_m,block_n,block_k,warps",
    [
        (16, 32, 32, 4),
        (32, 64, 64, 4),
        (64, 128, 128, 8),
    ],
)
def test_w8a16_graph_replay_consumes_new_activations(
    m, block_m, block_n, block_k, warps
):
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w8a16 import (
        _w8a16_gemm,
    )
    from vllm.triton_utils import triton

    torch.manual_seed(23)
    n, k = 70, 256
    quant = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int32)
    packed = torch.zeros(n, k // 4, device="cuda", dtype=torch.int32)
    for shift in range(4):
        packed |= (quant[:, shift::4] + 128) << (8 * shift)
    scales = (torch.rand(n, k // 128, device="cuda") * 0.001).bfloat16()
    weight = (quant.float() * scales.float().repeat_interleave(128, 1)).bfloat16()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=x.dtype)

    def call():
        _w8a16_gemm[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
            x,
            packed,
            scales,
            out,
            m,
            n,
            k,
            *x.stride(),
            GROUP_SIZE=128,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
        )

    call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        x.normal_()
        graph.replay()
        torch.testing.assert_close(
            out.float(), x.float() @ weight.float().T, atol=0.02, rtol=0.02
        )
