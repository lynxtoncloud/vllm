# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 loader contracts runnable without native extensions or GPU imports.

Run with --confcutdir=tests/standalone_tests. Execute the model's actual helper
and streaming-loader functions in isolation; full model tests run separately.
"""

import ast
import builtins
import runpy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm.scalar_type import scalar_types

DEVICE = "cpu"


def _wna16_debug():
    path = Path(__file__).parents[2] / (
        "vllm/model_executor/layers/fused_moe/wna16_debug.py"
    )
    return runpy.run_path(str(path))


def _tensor_pointer_utils():
    path = Path(__file__).parents[2] / "vllm/triton_utils/tensor_pointer.py"
    return runpy.run_path(str(path))


def _gemm2_call():
    return dict(
        A=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.bfloat16),
        B=torch.tensor([[[129, 130, 127, 126]]], dtype=torch.uint8),
        B_scale=torch.tensor([[[2, 0.5]]], dtype=torch.bfloat16),
        B_zp=None,
        C=torch.full((1, 2, 1), float("nan"), dtype=torch.bfloat16),
        topk_weights=torch.tensor([[0.5, 0.25]]),
        sorted_token_ids=torch.tensor([0, 2, 1, 2], dtype=torch.int32),
        expert_ids=torch.tensor([0, -1], dtype=torch.int32),
        num_tokens_post_padded=torch.tensor([4], dtype=torch.int32),
        mul_routed_weight=True,
        top_k=1,
        config={"BLOCK_SIZE_M": 2},
        block_shape=[0, 2],
    )


def test_wna16_replay_reference_dequantizes_groups_and_zeros_remote_experts():
    call = _gemm2_call()
    result = _wna16_debug()["reference"](call)
    # (1*2 + 2*4 - 3*0.5 - 4*1) * 0.5 = 2.25; remote row = 0.
    torch.testing.assert_close(result, torch.tensor([[[2.25], [0.0]]]).double())


@pytest.mark.parametrize("ids", [[0, 2, 0, 2], [0, 2, 2, 2], [-1, 2, 1, 2]])
def test_wna16_replay_rejects_invalid_assignment_instead_of_hiding_it(ids):
    call = _gemm2_call()
    call["sorted_token_ids"] = torch.tensor(ids, dtype=torch.int32)
    with pytest.raises(ValueError, match="Duplicate|Missing|Negative"):
        _wna16_debug()["reference"](call)


def test_wna16_failure_capture_is_failure_only_and_preserves_strides(tmp_path):
    debug = _wna16_debug()
    call = _gemm2_call()
    # Preserve a non-contiguous row layout across CPU capture and replay.
    call["A"] = torch.arange(16, dtype=torch.bfloat16).view(2, 8)[:, :4]
    path = debug["save_failure"](str(tmp_path), rank=4, module="mlp", **call)
    saved = torch.load(path, weights_only=True)
    call["A"].zero_()
    restored = debug["restore_call"](saved, "cpu")
    assert restored["A"].stride() == (8, 1)
    assert restored["A"][1, 0].item() == 8
    assert saved["rank"] == 4
    assert torch.isnan(restored["C"]).all()
    call["C"].zero_()
    assert debug["save_failure"](str(tmp_path), rank=4, module="mlp", **call) is None
    assert len(list(tmp_path.glob("*.pt"))) == 1


def _finite_checks():
    path = Path(__file__).parents[2] / "vllm/models/glm5next/diagnostics.py"
    return runpy.run_path(str(path))["install_finite_checks"]


def test_gemm2_shared_replay_preserves_values_strides_and_disjoint_output():
    call = _gemm2_call()
    call["A"] = torch.arange(16, dtype=torch.bfloat16).view(2, 8)[:, :4]
    share = _wna16_debug()["share_workspace"]
    shared = share(call, 4096)
    a, c = shared["A"], shared["C"]
    assert a.untyped_storage().data_ptr() == c.untyped_storage().data_ptr()
    assert a.untyped_storage().nbytes() == 4096
    assert c.storage_offset() * c.element_size() == 256
    assert a.stride() == call["A"].stride()
    torch.testing.assert_close(a, call["A"])
    assert torch.isnan(c).all()
    c.zero_()
    torch.testing.assert_close(a, call["A"])
    assert torch.isnan(call["C"]).all()
    assert shared["B"] is call["B"]
    with pytest.raises(ValueError, match="at least"):
        share(call, 256)


def test_replay_byte_audit_accepts_matching_nan_and_strided_values():
    same_bits = _wna16_debug()["same_bits"]
    value = torch.tensor([[float("nan"), 1, 2, 3]], dtype=torch.bfloat16)[:, ::2]
    assert same_bits(value, value.clone())
    changed = value.clone()
    changed[0, 1] = 4
    assert not same_bits(value, changed)


def test_replay_byte_audit_detects_differences_hidden_by_numeric_equality():
    same_bits = _wna16_debug()["same_bits"]
    assert not same_bits(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not same_bits(torch.tensor([1.0]), torch.tensor([1.0]).double())
    assert not same_bits(torch.tensor([1.0]), torch.tensor([[1.0]]))


def test_replay_view_pointer_preserves_address_and_bounds_strided_accesses():
    storage = torch.arange(1024, dtype=torch.bfloat16)
    view = storage[32:64].view(4, 8)[:, :3]
    pointer = _tensor_pointer_utils()["TensorView"](view)
    assert pointer.data_ptr() == view.data_ptr()
    assert pointer.size() == view.size()
    assert pointer.stride() == (8, 1)
    assert pointer.dtype == view.dtype
    assert pointer.ptr_range() == (3 * 8 + 3) * 2
    assert pointer.ptr_range() > view.numel() * view.element_size()
    assert pointer.ptr_range() < storage.untyped_storage().nbytes()
    view[0, 0] = 17
    assert pointer.tensor[0, 0].item() == 17


def test_pointer_bounds_only_specialize_small_views_of_large_storage():
    utils = _tensor_pointer_utils()
    bounded = utils["bounded_tensor_view"]
    # Meta tensors exercise the real storage-size boundary without allocating GiBs.
    small = torch.empty(2**31 - 1, dtype=torch.uint8, device="meta")
    small_view = small[:64]
    assert bounded(small_view) is small_view
    large = torch.empty(2**31 + 4096, dtype=torch.uint8, device="meta")
    view = large[32:96]
    assert bounded(view).tensor is view
    assert bounded(view).ptr_range() == 64
    assert bounded(large) is large
    # Two elements can span more than 2 GiB: numel() is not a sufficient bound.
    sparse_view = large.as_strided((2,), (2**31,))
    assert bounded(sparse_view) is sparse_view
    empty = large[:0]
    assert bounded(empty) is empty
    mock = NS(dtype=torch.bfloat16, ptr_range=lambda: 0)
    assert bounded(mock) is mock


@pytest.mark.parametrize("rocm,int8", [(True, True), (True, False), (False, True)])
def test_wna16_launcher_bounds_only_rocm_int8_workspace_views(rocm, int8):
    path = Path(__file__).parents[2] / (
        "vllm/model_executor/layers/fused_moe/fused_moe.py"
    )
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "invoke_fused_moe_wna16_triton_kernel"
    )
    captured = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: captured.extend(args)

    namespace = dict(
        torch=torch,
        Any=object,
        tl=NS(dtype=object),
        current_platform=NS(is_rocm=lambda: rocm),
        bounded_tensor_view=_tensor_pointer_utils()["bounded_tensor_view"],
        get_moe_wna16_block_config=lambda **kwargs: {
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
        },
        fused_moe_kernel_gptq_awq=Kernel(),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    storage = torch.empty(2**31 + 4096, dtype=torch.uint8, device="meta")
    call = _gemm2_call()
    call["A"] = storage[:16].view(torch.bfloat16).reshape(2, 4)
    call["C"] = storage[256:260].view(torch.bfloat16).reshape(1, 2, 1)
    namespace[fn.name](
        **call, compute_type=object(), use_int8_w8a16=int8, use_int4_w4a16=not int8
    )
    for index, name in ((0, "A"), (2, "C")):
        if rocm and int8:
            assert captured[index].tensor is call[name]
            assert captured[index].ptr_range() == call[name].numel() * 2
        else:
            assert captured[index] is call[name]
    assert captured[1] is call["B"]


@pytest.mark.parametrize("int8,isolate", [(True, True), (True, False), (False, True)])
def test_gemm2_isolation_preserves_layout_and_gemm1_storage(int8, isolate):
    path = Path(__file__).parents[2] / (
        "vllm/model_executor/layers/fused_moe/experts/triton_moe.py"
    )
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "TritonWNA16Experts"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_allocate_gemm2_output"
    )
    namespace = {
        "torch": torch,
        "_resize_cache": lambda cache, shape: cache[:16].view(shape),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    owner = NS(quant_config=NS(use_int8_w8a16=int8))
    if isolate:
        owner._diagnostic_isolate_gemm2 = True
    workspace = torch.ones(32, dtype=torch.bfloat16)
    output = namespace["_allocate_gemm2_output"](owner, workspace, (2, 2, 4))
    assert output.shape == (2, 2, 4) and output.stride() == (8, 4, 1)
    assert output.dtype == workspace.dtype and output.device == workspace.device
    output.zero_()
    assert workspace[:16].eq(1 if int8 and isolate else 0).all()
    assert workspace[16:].eq(1).all()


@pytest.mark.parametrize("source", ["input", "parameter", "output"])
def test_finite_diagnostics_identify_origin_before_downstream_execution(source):
    class Broken(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            if source == "output":
                return x / 0
            return x * self.scale

    layer = Broken()
    downstream = torch.nn.Identity()
    ran = Mock()
    downstream.register_forward_hook(ran)
    model = torch.nn.Sequential(layer, downstream)
    _finite_checks()(
        model, rank=9, enforce_eager=True, active=lambda: True, module_types=(Broken,)
    )
    x = torch.ones(2)
    if source == "input":
        x[0] = float("nan")
    elif source == "parameter":
        layer.scale.data.fill_(float("nan"))
    location = {
        "input": r"module=model input\[0\]",
        "parameter": "module=0 parameter=scale",
        "output": "module=0 output",
    }[source]
    with pytest.raises(RuntimeError, match=f"tp_rank=9 {location}"):
        model(x)
    ran.assert_not_called()


def test_finite_diagnostics_skip_profiling_and_unused_cache_but_check_real_kwargs():
    class Echo(torch.nn.Module):
        def forward(self, *, payload):
            return {"result": (payload["x"], None)}

    model = Echo()
    model.register_buffer("unused_cache", torch.tensor([float("nan")]))
    active = False
    _finite_checks()(
        model, rank=0, enforce_eager=True, active=lambda: active, module_types=()
    )
    x = torch.tensor([float("inf")])
    assert model(payload={"x": x})["result"][0] is x
    active = True
    with pytest.raises(RuntimeError, match=r"kwargs\[payload\]\[x\]"):
        model(payload={"x": x})
    x = torch.ones(2)
    assert model(payload={"x": x})["result"][0] is x
    torch.testing.assert_close(x, torch.ones(2))


def test_finite_diagnostics_reject_graph_mode_before_installing_hooks():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="requires --enforce-eager"):
        _finite_checks()(
            model, rank=0, enforce_eager=False, active=lambda: True, module_types=()
        )
    assert not model._forward_hooks and not model._forward_pre_hooks


@pytest.mark.parametrize(
    "fault,location",
    [
        ("parameter", "routed_experts parameter=scale"),
        ("router", "router stage=select_experts output"),
        ("experts", "routed_experts stage=forward_modular output"),
        ("gemm1", "stage=gemm1 output"),
        ("activation", "stage=activation output"),
        ("gemm2", "stage=gemm2 output"),
        ("moe_sum", "stage=moe_sum output"),
        ("before_reduce", "stage=_maybe_reduce_final_output input"),
        ("reduce", "stage=_maybe_reduce_final_output output"),
        (None, None),
    ],
)
def test_finite_moe_diagnostics_separate_local_computation_and_reduction(
    fault, location
):
    class TritonWNA16Experts:
        def activation(self, activation, output, input):
            output.copy_(input)
            if fault == "activation":
                output.fill_(float("nan"))

        def moe_sum(self, input, output):
            output.copy_(input)
            if fault == "moe_sum":
                output.fill_(float("inf"))

    class Routed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            value = float("nan") if fault == "parameter" else 1.0
            self.scale = torch.nn.Parameter(torch.tensor(value))
            self.quant_method = NS(moe_kernel=NS(fused_experts=TritonWNA16Experts()))

        def forward_modular(self, *, x, **kwargs):
            if fault == "experts":
                return x / 0
            experts = self.quant_method.moe_kernel.fused_experts
            gemm1 = x / 0 if fault == "gemm1" else x * self.scale
            # In-place outputs are undefined until their producer has run.
            activated = torch.full_like(x, float("nan"))
            experts.activation(None, activated, gemm1)
            gemm2 = activated / 0 if fault == "gemm2" else activated
            output = torch.full_like(x, float("nan"))
            experts.moe_sum(gemm2, output)
            return output

    class Runner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.routed_experts = Routed()
            weight = float("nan") if fault == "router" else 1.0
            self.router = NS(
                select_experts=lambda: (torch.tensor([weight]), torch.tensor([0]))
            )

        def _apply_quant_method(self, x):
            weights, ids = self.router.select_experts()
            return None, self.routed_experts.forward_modular(
                x=x, topk_weights=weights, topk_ids=ids
            )

        def _maybe_reduce_final_output(self, x):
            return x / 0 if fault == "reduce" else x * 2

        def forward(self, x):
            _, result = self._apply_quant_method(x)
            if fault == "before_reduce":
                result = result / 0
            return self._maybe_reduce_final_output(result)

    model = torch.nn.Sequential(Runner())
    _finite_checks()(
        model, rank=1, enforce_eager=True, active=lambda: True, module_types=()
    )
    if location is None:
        torch.testing.assert_close(model(torch.ones(2)), torch.full((2,), 2.0))
    else:
        with pytest.raises(RuntimeError, match=location):
            model(torch.ones(2))


def _model_functions():
    path = Path(__file__).parents[2] / "vllm/models/glm5next/nvidia/model.py"
    names = {"_try_load_w8a16_attention_weight", "_is_w8a16_g128"}
    functions: list[ast.stmt] = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch, "QuantizationConfig": object}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


helpers = _model_functions()


@pytest.mark.parametrize("rope_dim,has_sinks", [(0, False), (0, True), (64, True)])
def test_rocm_sparse_metadata_without_persistent_kernel_needs_no_aiter(
    monkeypatch, rope_dim, has_sinks
):
    """Construct real metadata buffers while forbidding the optional dependency."""
    path = Path(__file__).parents[2] / (
        "vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ROCMAiterMLASparseMetadataBuilder"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    select = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_use_rocm_sparse_triton"
    )
    ns: dict = dict(
        torch=torch,
        AttentionSpec=object,
        VllmConfig=object,
        get_mla_dims=lambda _: NS(kv_lora_rank=512, qk_rope_head_dim=rope_dim),
        current_platform=NS(num_compute_units=lambda: 96),
    )
    exec(
        compile(ast.Module(body=[select, init], type_ignores=[]), str(path), "exec"), ns
    )
    original_import = builtins.__import__

    def no_aiter(name, *args, **kwargs):
        if name == "aiter" or name.startswith("aiter."):
            pytest.fail("Nonpersistent metadata must not import AITER")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_aiter)
    cfg = NS(
        model_config=NS(
            dtype=torch.bfloat16,
            max_model_len=4096,
            hf_text_config=NS(index_topk=2048),
            get_num_attention_heads=lambda _: 16,
        ),
        cache_config=NS(cache_dtype="auto"),
        parallel_config=NS(),
        scheduler_config=NS(max_num_batched_tokens=4),
        compilation_config=NS(
            static_forward_context={
                "layer": NS(impl=NS(sinks=torch.zeros(16) if has_sinks else None))
            }
        ),
    )
    instance = NS(_init_reorder_batch_threshold=Mock())
    ns["__init__"](instance, NS(block_size=16), ["layer"], cfg, torch.device("cpu"))
    assert instance._use_persistent_metadata is False
    assert instance._prev_metadata_key is None
    assert instance.paged_kv_indptr.shape == (5,)
    assert instance.paged_kv_indices.shape == (4 * 2048,)


def _reblock_indexer_table():
    path = Path(__file__).parents[2] / "vllm/v1/attention/backends/mla/indexer.py"
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_reblock_indexer_table"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["_reblock_indexer_table"]


@pytest.mark.parametrize("source,target", [(640, 128), (128, 640), (128, 128)])
def test_indexer_reblocking_preserves_physical_token_addresses(source, target):
    """Prefill writes and decode reads must address the same physical tokens."""
    coarse = max(source, target)
    blocks = torch.tensor([[3, 1, 7], [9, 0, 2]], dtype=torch.int32)
    factor = coarse // source
    source_table = (
        blocks.unsqueeze(-1) * factor + torch.arange(factor, dtype=torch.int32)
    ).flatten(1)
    target_table = _reblock_indexer_table()(source_table, source, target)
    positions = torch.arange(3 * coarse)
    expected = source_table[:, positions // source] * source + positions % source
    actual = target_table[:, positions // target] * target + positions % target
    torch.testing.assert_close(actual, expected)


def test_indexer_reblocking_preserves_invalid_pages():
    table = torch.tensor([[2, -1]], dtype=torch.int32)
    expanded = _reblock_indexer_table()(table, 640, 128)
    torch.testing.assert_close(
        expanded,
        torch.tensor([[10, 11, 12, 13, 14, -1, -1, -1, -1, -1]]).to(torch.int32),
    )


@pytest.mark.parametrize(
    "block_size,pages_per_block", [(1, 1), (16, 1), (32, 1), (64, 1), (32, 5)]
)
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("per_query_lengths", [False, True])
def test_rocm_decode_logits_read_preshuffled_cache(
    block_size, pages_per_block, next_n, per_query_lengths
):
    """Decode and MTP must agree with logical K despite physical page shuffling."""
    path = Path(__file__).parents[2] / (
        "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
    )
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "fp8_paged_mqa_logits_torch"
    )
    ns: dict = dict(
        torch=torch,
        F=torch.nn.functional,
        current_platform=NS(fp8_dtype=lambda: torch.float8_e4m3fn),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    torch.manual_seed(71)
    dim, heads = 128, 2
    # Reproduce the 32K decode failure: 8192 pooled states need 256 small
    # pages, but the common table has only 205 entries of 640 raw tokens.
    num_blocks = 205 if pages_per_block > 1 else 4
    num_pages = num_blocks * pages_per_block
    capacity = num_pages * block_size
    length = 8192 if pages_per_block > 1 else capacity - 1
    keys = torch.randn(capacity, dim).to(torch.float8_e4m3fn)
    scales = torch.rand(capacity) + 0.1
    cache = torch.empty(num_pages, block_size * (dim + 4), dtype=torch.uint8)
    token, d = torch.arange(block_size)[:, None], torch.arange(dim)[None, :]
    offsets = token * dim + d
    if block_size > 1:
        offsets = token // 16 * 16 * dim + d // 16 * 256 + token % 16 * 16 + d % 16
    # Reverse physical pages so the reference also checks the block table.
    common_table = torch.arange(num_blocks - 1, -1, -1, dtype=torch.int32)
    table = (
        common_table[:, None] * pages_per_block
        + torch.arange(pages_per_block, dtype=torch.int32)
    ).flatten()
    for logical, physical in enumerate(table):
        sl = slice(logical * block_size, (logical + 1) * block_size)
        cache[physical, offsets] = keys[sl].view(torch.uint8)
        cache[physical, block_size * dim :] = scales[sl].view(torch.uint8)
    query = torch.randn(2, next_n, heads, dim).to(torch.float8_e4m3fn)
    weights = torch.randn(2 * next_n, heads)
    limits = torch.arange(length - next_n + 1, length + 1)
    lens = (
        torch.stack((limits, torch.zeros_like(limits)))
        if per_query_lengths
        else torch.tensor([length, 0])
    )
    actual = ns["fp8_paged_mqa_logits_torch"](
        query,
        cache.view(num_pages, block_size, 1, dim + 4),
        weights,
        lens.to(torch.int32),
        _reblock_indexer_table()(
            common_table.repeat(2, 1),
            block_size * 4 * pages_per_block,
            block_size * 4,
        ),
        capacity,
    )
    expected = torch.full((2 * next_n, capacity), float("-inf"))
    for j, limit in enumerate(limits):
        scores = (query[0, j].float() @ keys[:limit].float().T).relu()
        expected[j, :limit] = (scores * weights[j, :, None]).sum(0) * scales[:limit]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("rows", [0, 1, 9])
def test_rocm_prefill_logits_bound_head_workspace(monkeypatch, rows):
    """Chunking must preserve logits/masking without an H*M*N workspace."""
    path = Path(__file__).parents[2] / (
        "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
    )
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "fp8_mqa_logits_torch"
    )
    heads, keys, dim = 2, 31, 128
    budget = heads * 3 * keys * 4
    ns: dict = dict(torch=torch, _TORCH_MQA_LOGITS_MAX_SCORE_BYTES=budget)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    torch.manual_seed(15)
    q = torch.randn(rows, heads, dim).to(torch.float8_e4m3fn)
    k = torch.randn(keys, dim).to(torch.float8_e4m3fn)
    scales = torch.rand(keys, 1) + 0.1
    weights = torch.randn(rows, heads)
    starts = torch.arange(rows) % keys
    ends = torch.clamp(starts + 8, max=keys)
    score = torch.einsum("mhd,nd->hmn", q.bfloat16(), k.bfloat16()).float()
    expected = ((score * scales.flatten()).relu() * weights.T[:, :, None]).sum(0)
    columns = torch.arange(keys)[None, :]
    expected.masked_fill_(
        (columns < starts[:, None]) | (columns >= ends[:, None]), float("-inf")
    )
    original_einsum = torch.einsum

    def bounded_einsum(*args):
        result = original_einsum(*args)
        assert result.numel() * 4 <= budget
        return result

    monkeypatch.setattr(torch, "einsum", bounded_einsum)
    actual = ns["fp8_mqa_logits_torch"](q, (k, scales), weights, starts, ends)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("index_kpool", [1, 4])
@pytest.mark.parametrize("aiter_enabled", [False, True])
def test_rocm_kpool_dispatch_does_not_require_aiter(index_kpool, aiter_enabled):
    """The pooled indexer must reach profiling/fallbacks with AITER disabled."""
    path = Path(__file__).parents[2] / (
        "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
    )
    method = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "forward_hip"
    )
    native = Mock(return_value="native")
    enabled = Mock(return_value=aiter_enabled)
    namespace: dict = dict(
        torch=NS(
            Tensor=torch.Tensor, ops=NS(vllm=NS(rocm_aiter_sparse_attn_indexer=native))
        ),
        rocm_aiter_ops=NS(is_enabled=enabled),
        _encode_layer_name=lambda name: name,
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    shared = Mock(return_value="pooled")
    instance = NS(
        use_fp4_cache=False,
        forward_cuda=shared,
        k_cache=NS(prefix="indexer", kv_cache=None),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=2048,
        head_dim=128,
        max_model_len=32768,
        max_total_seq_len=32768,
        topk_indices_buffer=None,
        skip_k_cache_insert=False,
    )
    x = torch.empty(0)
    kwargs = dict(gate_score=x, compress_ape=x, index_kpool=index_kpool, positions=x)
    if index_kpool > 1:
        assert namespace["forward_hip"](instance, x, x, x, x, **kwargs) == "pooled"
        shared.assert_called_once_with(x, x, x, x, **kwargs)
        enabled.assert_not_called()
        native.assert_not_called()
    elif aiter_enabled:
        assert namespace["forward_hip"](instance, x, x, x, x, **kwargs) == "native"
        native.assert_called_once()
        shared.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="only supported on AITER"):
            namespace["forward_hip"](instance, x, x, x, x, **kwargs)
        shared.assert_not_called()
        native.assert_not_called()


def w8a16_config():
    return NS(
        get_name=lambda: "compressed-tensors",
        quant_format="pack-quantized",
        target_scheme_map={
            "Linear": {
                "weights": NS(
                    num_bits=8,
                    type="int",
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                    dynamic=False,
                    actorder=None,
                ),
                "input_activations": None,
                "output_activations": None,
            }
        },
    )


def test_w8a16_detection_depends_only_on_checkpoint_format():
    assert helpers["_is_w8a16_g128"](w8a16_config())
    assert not helpers["_is_w8a16_g128"](None)
    assert not helpers["_is_w8a16_g128"](NS(get_name=lambda: "fp8"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_bits", 4),
        ("type", "float"),
        ("symmetric", False),
        ("strategy", "channel"),
        ("group_size", 64),
        ("dynamic", True),
        ("actorder", "group"),
    ],
)
def test_other_weight_schemes_do_not_enter_w8a16_loader(field, value):
    config = w8a16_config()
    setattr(config.target_scheme_map["Linear"]["weights"], field, value)
    assert not helpers["_is_w8a16_g128"](config)


@pytest.mark.parametrize(
    "group_size,k,dtype,zero_points,expected",
    [
        (128, 256, torch.bfloat16, False, True),
        (128, 256, torch.float16, False, True),
        (64, 256, torch.bfloat16, False, False),
        (128, 192, torch.bfloat16, False, False),
        (128, 256, torch.float32, False, False),
        (128, 256, torch.bfloat16, True, False),
    ],
)
def test_w8a16_linear_dispatch_rejects_unsupported_partitions(
    group_size, k, dtype, zero_points, expected
):
    """Run the actual W8 selector and ROCm registry without native imports."""
    root = Path(__file__).parents[2] / "vllm/model_executor/kernels/linear"
    tree = ast.parse((root / "mixed_precision/triton_w8a16.py").read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = dict(
        **helpers,
        MPLinearKernel=object,
        MPLinearLayerConfig=NS,
        scalar_types=scalar_types,
        current_platform=NS(is_rocm=lambda: True),
    )
    exec(
        compile(ast.Module(body=[kernel], type_ignores=[]), "w8a16", "exec"), namespace
    )
    implementation = namespace["TritonW8A16LinearKernel"]
    registry = next(
        node
        for node in ast.parse((root / "__init__.py").read_text()).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_POSSIBLE_KERNELS"
    )
    assert isinstance(registry.value, ast.Dict)
    rocm = next(
        value
        for key, value in zip(registry.value.keys, registry.value.values)
        if isinstance(key, ast.Attribute) and key.attr == "ROCM"
    )
    assert isinstance(rocm, ast.List)
    assert any(
        isinstance(node, ast.Name) and node.id == implementation.__name__
        for node in rocm.elts
    )
    cfg = NS(
        partition_weight_shape=(k, 35),
        full_weight_shape=(512, 70),
        weight_type=scalar_types.uint8b128,
        act_type=dtype,
        out_type=None,
        group_size=group_size,
        zero_points=zero_points,
    )
    supported, reason = implementation.can_implement(cfg)
    assert supported is expected
    assert (reason is None) is expected


@pytest.mark.parametrize(
    "source,target,shard",
    [
        ("q_a_proj", "fused_qkv_a_proj", 0),
        ("kv_a_proj_with_mqa", "fused_qkv_a_proj", 1),
        ("indexer.wk", "indexer.wk_weights_proj", 0),
        ("indexer.weights_proj", "indexer.wk_weights_proj", 1),
        ("q_proj", "in_proj_qkvbfg_a", 0),
        ("k_proj", "in_proj_qkvbfg_a", 1),
        ("v_proj", "in_proj_qkvbfg_a", 2),
        ("b_proj", "in_proj_qkvbfg_a", 3),
        ("f_a_proj", "in_proj_qkvbfg_a", 4),
        ("g_a_proj", "in_proj_qkvbfg_a", 5),
        ("f_b_proj", "f_b_proj", None),
        ("g_b_proj", "g_b_proj", None),
        ("q_b_proj", "q_b_proj", None),
        ("kv_b_proj", "kv_b_proj", None),
        ("o_proj", "o_proj", None),
    ],
)
@pytest.mark.parametrize("scale_first", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_w8a16_attention_loading_preserves_offset_groups_and_shards(
    source, target, shard, scale_first, dtype
):
    """Use every byte value and distinct scales across groups and output rows."""
    base = "layers.0.self_attn."
    q = torch.arange(256, device=DEVICE).repeat(3, 1).int() - 128
    packed = torch.zeros((3, 64), dtype=torch.int32, device=DEVICE)
    for i in range(4):
        packed |= (q[:, i::4] + 128) << (8 * i)
    scales = torch.tensor(
        [[0.125, 0.5], [1.0, 2.0], [0.25, 4.0]], dtype=torch.bfloat16, device=DEVICE
    )
    param = torch.nn.Parameter(torch.empty(0, dtype=dtype, device=DEVICE))
    param.weight_loader = Mock()
    target_name = base + target + ".weight"
    fields = [
        ("weight_packed", packed),
        ("weight_scale", scales),
        ("weight_shape", torch.tensor([3, 256], device=DEVICE)),
    ]
    if scale_first:
        fields.reverse()
    pending: dict = {}
    loaded: set[str] = set()
    mapping = [
        (".fused_qkv_a_proj", ".q_a_proj", 0),
        (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
        (".wk_weights_proj", ".wk", 0),
        (".wk_weights_proj", ".weights_proj", 1),
        *[
            (".in_proj_qkvbfg_a", f".{proj}_proj", shard_id)
            for shard_id, proj in enumerate(["q", "k", "v", "b", "f_a", "g_a"])
        ],
    ]
    for field, tensor in fields:
        assert helpers["_try_load_w8a16_attention_weight"](
            base + source + "." + field,
            tensor,
            pending,
            {target_name: param},
            loaded,
            mapping,
        )
    assert not pending and loaded == {target_name}
    args = param.weight_loader.call_args.args
    assert len(args) == (2 if shard is None else 3)
    if shard is not None:
        assert args[2] == shard
    expected = (q.float() * scales.float().repeat_interleave(128, 1)).to(dtype)
    torch.testing.assert_close(args[1], expected, rtol=0, atol=0)


def test_w8a16_loader_leaves_quantized_parameters_for_normal_loading():
    name = "layers.0.self_attn.q_b_proj.weight_packed"
    param = torch.nn.Parameter(
        torch.zeros((2, 32), dtype=torch.int32), requires_grad=False
    )
    pending: dict = {}
    loaded: set[str] = set()
    assert not helpers["_try_load_w8a16_attention_weight"](
        name,
        param,
        pending,
        {name: param},
        loaded,
        [],
    )
    assert not pending and not loaded


@pytest.mark.parametrize("complete", [True, False])
@pytest.mark.parametrize(
    "source,target,shard_id",
    [
        ("q_a_proj", "fused_qkv_a_proj", 0),
        *[
            (f"{proj}_proj", "in_proj_qkvbfg_a", shard)
            for shard, proj in enumerate(["q", "k", "v", "b", "f_a", "g_a"])
        ],
    ],
)
def test_glm_weight_loader_consumes_packed_attention_or_rejects_missing_scale(
    complete, source, target, shard_id
):
    """Execute the actual streaming loader without constructing the GPU model."""
    path = Path(__file__).parents[2] / "vllm/models/glm5next/nvidia/model.py"
    model = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextModel"
    )
    loader = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    namespace = dict(
        **helpers,
        Iterable=list,
        get_spec_layer_idx_from_weight_name=lambda *_: None,
        _try_load_fp8_indexer_wk=lambda *_: False,
        _try_load_fp8_attn_proj=lambda *_: False,
    )
    exec(
        compile(ast.Module(body=[loader], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    param = torch.nn.Parameter(torch.zeros((1, 128), dtype=torch.bfloat16))

    def weight_loader(p, w, shard):
        assert shard == shard_id
        p.data.copy_(w)

    param.weight_loader = weight_loader
    target = f"layers.0.self_attn.{target}.weight"
    instance = NS(
        config=NS(is_moe=False, mla_nope=False),
        quant_config=w8a16_config(),
        named_parameters=lambda: [(target, param)],
    )
    base = f"layers.0.self_attn.{source}."
    weights = [
        (base + "weight_packed", torch.full((1, 32), -2122219135, dtype=torch.int32)),
        (base + "weight_shape", torch.tensor([1, 128])),
    ]
    if complete:
        # 0x81818181 decodes to +1 in every position (offset binary).
        weights.append(
            (base + "weight_scale", torch.ones((1, 1), dtype=torch.bfloat16))
        )
        assert namespace["load_weights"](instance, iter(weights)) == {target}
        torch.testing.assert_close(param, torch.ones_like(param))
    else:
        with pytest.raises(ValueError, match="Incomplete W8A16"):
            namespace["load_weights"](instance, iter(weights))


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_mtp_w8a16_loader_rewrites_prefix_and_loads_attention(multimodal, complete):
    """MTP must consume packed attention after its checkpoint-name rewrite."""
    path = Path(__file__).parents[2] / "vllm/models/glm5next/nvidia/mtp.py"
    model = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextMTP"
    )
    methods: list[ast.stmt] = [
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("load_weights", "_rewrite_spec_layer_name")
    ]
    namespace = dict(
        **helpers,
        Iterable=list,
        fused_moe_make_expert_params_mapping=lambda *args, **kwargs: [],
        get_spec_layer_idx_from_weight_name=lambda cfg, name: (
            32 if ".layers.32." in name else None
        ),
        _try_load_fp8_indexer_wk=lambda *_: False,
        _try_load_fp8_attn_proj=lambda *_: False,
    )
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    param = torch.nn.Parameter(torch.zeros((1, 128), dtype=torch.bfloat16))

    def weight_loader(p, w, shard):
        assert shard == 0
        p.data.copy_(w)

    param.weight_loader = weight_loader
    target = "model.layers.32.mtp_block.self_attn.fused_qkv_a_proj.weight"
    instance = NS(
        config=NS(mla_nope=False, n_routed_experts=1),
        quant_config=w8a16_config(),
        num_redundant_experts=0,
        model=NS(mtp_start_layer_idx=32, num_mtp_layers=1),
        named_parameters=lambda: [(target, param)],
    )
    instance._rewrite_spec_layer_name = lambda idx, name: namespace[
        "_rewrite_spec_layer_name"
    ](instance, idx, name)
    prefix = "model.language_model" if multimodal else "model"
    base = f"{prefix}.layers.32.self_attn.q_a_proj."
    weights = [
        (base + "weight_shape", torch.tensor([1, 128])),
        (base + "weight_packed", torch.full((1, 32), -2122219135, dtype=torch.int32)),
    ]
    if complete:
        weights.append(
            (base + "weight_scale", torch.ones((1, 1), dtype=torch.bfloat16))
        )
        assert namespace["load_weights"](instance, iter(weights)) == {target}
        torch.testing.assert_close(param, torch.ones_like(param))
    else:
        with pytest.raises(ValueError, match="Incomplete W8A16"):
            namespace["load_weights"](instance, iter(weights))
