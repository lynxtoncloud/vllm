# Long-context input preparation diagnostics

`VLLM_TRACE_MODEL_INPUTS=1` adds device fences around V2 input preparation.
It requires eager execution and remains inactive until worker startup warmup
has completed. Dummy, profiling and zero-token batches are excluded. It does
not change model arithmetic, context capacity, batching or KV layout.

Use this when an asynchronous ROCm memory-access error surfaces at an unrelated
later API call. Global `AMD_SERIALIZE_KERNEL=3` changes communication-kernel
scheduling as well; if it prevents distributed startup, remove it and use these
request-stage fences. A successful fenced run can mask timing-sensitive bugs
and does not establish a production fix or performance result.

For the saved 512K diagnostic run, update source on P0/D0/D1 to the same commit
before invoking the controller on **P0**:

```bash
cd /data/vllm
base=/data/logs/glm53-assessment/diag512k-input-$(date +%Y%m%d-%H%M%S)
mkdir -p "$base"
nohup .venv/bin/python -u \
  examples/disaggregated/glm5next_int8/diagnose_input_preparation.py \
  --source-dir /data/logs/glm53-assessment/diag512k-sync-20260916-222356 \
  --output "$base" > "$base/console.log" 2>&1 < /dev/null &
echo "$!" > "$base/pid"
echo "$base" > /data/logs/glm53-assessment/latest-diag512k-input-dir.txt
```

The source directory must contain `config-d.json`, `request.json`, and
`expected.json`. The controller validates these before stopping anything. It
stops the source `run.py` only when the saved PID still matches that exact script,
stops and cleans managed D0/D1 workers, and starts D0/D1 under the new run ID.
P0/P1 engines, the proxy and monitors remain running. The new D configuration
preserves the source workload settings, disables global launch blocking and
enables `NCCL_DEBUG=INFO` plus input tracing.

The controller waits for readiness and replays the saved token-ID request through
the existing proxy. It writes `status.json` and `response.json`. A passing result
requires completed streaming output, the expected input-token count, all three
needle answers, and KV-transfer metric checks. The request is saved unchanged
except that streaming and usage reporting are enabled.

Engine logs stay on their respective D hosts under
`/data/logs/glm53-assessment/<new-run-id>/<d0-or-d1>/engine/console.log`.
Filter for `Input preparation trace:`. Each JSON record contains rank, PID,
step, time, stage and event. `BEFORE` without `DONE`, or `FAILED`, identifies a
failing synchronization interval, not necessarily one individual kernel.
The entry fence can report faults left by earlier forward/sampling/transfer work.
All metadata printed before a fence is CPU data or tensor shape metadata.

For a P-side image or video failure, start both P0 and P1 with
`ENFORCE_EAGER=1 VLLM_TRACE_MODEL_INPUTS=1` and keep multimodal encoder
compilation and CUDA graphs disabled. Each rank logs the request IDs and
`BEGIN`/`READY`/`EXECUTED`/`DONE`/`FAILED` events for multimodal input preparation, encoder
execution, embedding collection, and embedding merge. GLM-5.3 vision logs
patch embedding, metadata, each numbered block, attention kernels, TP
projections, and the merger. The merger is split into `vision_downsample`,
with `vision_downsample_norm`, `vision_downsample_layout`, and
`vision_downsample_conv` nested inside it, followed by
`vision_merger_gather_projection`, `vision_merger_norm_activation`,
`vision_merger_gate_up_projection`, `vision_merger_activation`, and
`vision_merger_reduce_projection`. The gather and reduce projection spans
include both the matrix operation and any TP communication performed by the
parallel linear layer. A completed span includes elapsed milliseconds
and free, total, allocated, and reserved GPU memory before and after it.
Memory sampling failures appear as an `error` field and do not abort inference.
`ready_ms`, `body_ms`, and `sync_ms` split the completed span into its
entry fence, model call, and exit fence.
The convolution span also records its input shape, strides, and dtype. A long
`body_ms` identifies a slow synchronous call within that span; it does not by
itself distinguish library setup from GPU execution.

Compare the last `BEGIN` without a matching `DONE` on every rank. `BEGIN`
without `READY` points to the entry fence or memory sampling; `READY` without
`EXECUTED` points to the wrapped call; `EXECUTED` without `DONE` points to the
exit fence. A stalled
`vision_attention_tp_projection` or `vision_mlp_tp_projection` includes the
projection and its possible TP collective; the trace alone cannot distinguish
the two kernels. A `BEGIN` with no `DONE` can also mean a synchronization fence
is waiting for earlier asynchronous GPU work. Disable tracing and restart the
workers after collecting the diagnostic logs because per-stage fences affect
timing and throughput.

This controller leaves D running after the replay for inspection. Before a new
attempt, stop it using the managed D0/D1 run ID. To return to performance testing,
restart D with tracing disabled and restore the original logging configuration.

## P-side downsample profiling

To identify GPU kernels, collect a separate PyTorch profiler run with input
tracing disabled. When starting engines through `assessment_jobs.py`, make a
separate copy of `assessment_config.json` and merge these fields into its
`environment` object:

```json
{
  "VLLM_TRACE_MODEL_INPUTS": "0",
  "PROFILE_CONFIG": "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/data/logs/glm53-profile\",\"torch_profiler_with_stack\":false,\"torch_profiler_dump_cuda_time_total\":false}"
}
```

Keep `ENFORCE_EAGER` set to `"1"`. Create `/data/logs/glm53-profile` on all four
nodes before starting engines; the config is sent to every node, although only
the P engine is profiled below. Use a new run ID on P0:

```bash
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  start engines --run-id "$GLM_ASSESS_RUN" \
  --config /path/to/profile-config.json
```

Setting `PROFILE_CONFIG` only in the P0 shell does not pass it through SSH to
P1. The default assessment config already enables input tracing; leave it
unchanged for the split-stage log run.

After both P nodes are ready, start profiling through the P0 vLLM API, send
one image request through the proxy, then stop profiling to flush trace files:

```bash
curl --noproxy '*' -fsS -X POST http://10.5.10.36:8001/start_profile
curl --noproxy '*' --fail-with-body --max-time 600 \
  -H 'Content-Type: application/json' \
  --data-binary @/path/to/image.request.json \
  -o /tmp/image.response.json \
  http://10.5.10.36:8000/v1/chat/completions
curl --noproxy '*' -fsS --max-time 1800 -X POST \
  http://10.5.10.36:8001/stop_profile
```

Inspect `/data/logs/glm53-profile` separately on P0 and P1. The profiler
records CPU operator calls and GPU kernels, while the split input trace shows
which downsample sub-stage is slow. Do not compare profiled or synchronized
latencies with normal serving performance.
