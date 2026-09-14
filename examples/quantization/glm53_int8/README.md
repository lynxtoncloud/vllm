# GLM-5.3-Flash INT8 weights

This change enables the workspace converter's W8A16-G128 checkpoint in
GLM-5.3-Flash. It builds on main at `2e9f7bb45`, which already contains
[PR #53906](https://github.com/vllm-project/vllm/pull/53906) and subsequent fixes.
The existing BF16 checkout is unchanged.

## Checkpoint contract

Use the output of `deployment/glm53/fp8_to_int8.py`: compressed-tensors
`pack-quantized`, symmetric INT8 weights, groups of 128 along the input
axis, and no activation quantization. Four offset-binary values are packed
into each INT32; decoding subtracts 128 and multiplies by the group scale.
The converter preserves the tokenizer, processors, templates and vision weights.

## Implementation

- Main model and MTP loaders identify the checkpoint from its quantization
  configuration. Loading does not depend on a GPU architecture or environment
  flag. Other checkpoint formats continue through their existing loaders.
- MLA, indexer and KDA projections retain the upstream unquantized runtime
  representation. Converted projection tensors are dequantized during loading
  and passed to the original loaders for fusion and TP slicing. Destination
  BF16 and FP16 parameters are supported. KDA's f_a/g_a replicated shards retain
  their original shard IDs and loading behavior.
- Text, multimodal and MTP model classes expose gate/up fusion mappings for
  exact checkpoint targets. The multimodal mapping also retains vision QKV.
  MTP maps multimodal checkpoint prefixes to its text-model prefixes.
- Dense/shared MLPs and routed experts retain quantized weights through the
  existing compressed-tensors schemes and platform kernel selection.
- The added ROCm `TritonW8A16LinearKernel` fills the grouped INT8 dense GEMM gap.
  It is registered for ROCm, with no gfx1100-only check, and reads packed
  INT32 storage directly. It forms BF16/FP16 operands with FP32 accumulation
  without keeping a complete dequantized MLP weight copy. CUDA and other
  platforms retain their existing backend selection and support checks.

The change adds no custom MoE routing, cache, MHC, sparse-attention or NIXL
fallbacks. It does not override eager/graph mode, speculative decoding,
context length, concurrency, KV dtype, or multimodal input limits. Those remain
subject to normal model/platform support and vLLM validation. Removing the
previous experimental profile does not establish support on every GPU.

## Usage

Start with the usual GLM-5.3-Flash configuration for the target platform and
point it at the converted model directory, for example:

```bash
vllm serve /path/to/GLM-5.3-Flash-W8A16-G128 \
  --quantization compressed-tensors \
  --dtype bfloat16
```

Choose TP/EP, context length and batching for the model dimensions and available
memory. The new dense GEMM requires complete groups of 128 on each TP rank.
No extra gfx1100 profile or forced MoE backend is needed by this adaptation.

## Validation

```bash
.venv/bin/python -m pytest tests/standalone_tests/test_glm5next_int8.py \
  --confcutdir=tests/standalone_tests -q
.venv/bin/python -m pytest tests/quantization/test_compressed_tensors.py \
  -k glm5next_w8a16 -q
.venv/bin/python -m pytest tests/quantization/test_moe_wna16.py -q
.venv/bin/python -m pytest tests/kernels/quantization/test_triton_w8a16.py -q
.venv/bin/python -m pytest tests/kernels/moe/test_moe.py -k fused_moe_wn16 -q
```

On the local Mac, 93 standalone CPU tests pass. They execute the actual model
loading functions in isolation, covering format detection, packed-byte decoding,
BF16/FP16 destinations, projection shard routing, incomplete weight errors,
and MTP checkpoint-name rewriting. They do not instantiate the full model.

The full compressed-tensors and MoE suites require a complete vLLM runtime.
The new GPU GEMM tests require ROCm and cover row/column TP, separate gate/up
shards, bias, empty inputs, partial tiles and multiple scale groups. These GPU
and integration suites have not been executed here.

GPU compilation and full-model evaluation remain outstanding. Compare generation
and logprobs against a reference using the same dequantized W8 weights, then
measure quantization quality against the original checkpoint. Check image/video
and MTP separately on supported platforms. No server was deployed by this change.
