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
`vision_merger_gather_projection`, `vision_merger_norm_activation`,
`vision_merger_gate_up_projection`, `vision_merger_activation`, and
`vision_merger_reduce_projection`. The gather and reduce projection spans
include both the matrix operation and any TP communication performed by the
parallel linear layer. A completed span includes elapsed milliseconds
and free, total, allocated, and reserved GPU memory before and after it.
Memory sampling failures appear as an `error` field and do not abort inference.
`ready_ms`, `body_ms`, and `sync_ms` split the completed span into its
entry fence, model call, and exit fence.

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
