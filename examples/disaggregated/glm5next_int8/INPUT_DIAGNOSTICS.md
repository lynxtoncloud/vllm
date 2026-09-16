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

This controller leaves D running after the replay for inspection. Before a new
attempt, stop it using the managed D0/D1 run ID. To return to performance testing,
restart D with tracing disabled and restore the original logging configuration.
