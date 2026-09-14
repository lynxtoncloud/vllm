# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 loader contracts runnable without native extensions or GPU imports.

Run with --confcutdir=tests/standalone_tests. Execute the model's actual helper
and streaming-loader functions in isolation; full model tests run separately.
"""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm.scalar_type import scalar_types

DEVICE = "cpu"


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
