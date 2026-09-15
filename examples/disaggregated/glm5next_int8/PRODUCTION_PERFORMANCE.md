# GLM W8A16 production performance qualification

## Scope and completion criteria

This campaign evaluates P0+P1 (TP16/EP16 prefill) → D0+D1 (TP16/EP16 decode).
All administration runs on P0, using the existing clones and detached Python
supervisors. No systemd, new server worktree, context reduction, or model change
is required. The current large-image hang investigation is deferred.

Code availability is not hardware qualification. The new Triton indexer is an
explicit opt-in until the GPU regression suite and serving A/B results pass.
No tuned GPU parameters or performance improvements are claimed without results.

| Work | Implementation | Required evidence |
| --- | --- | --- |
| Indexer | Batched GPU paged/prefill logits, no host length reads, no full head×query×key intermediate | Torch comparison, ragged/page-padding cases, replay with changed inputs, serving correctness |
| Graph | Configurable PIECEWISE capture sizes; eager comparison retained | Real multi-request PD replay, throughput/TTFT/TPOT, graph memory, no stale KV |
| W8A16 linear | Offline tuning, exact-shape/device config loading | Numerical checks per candidate, measured timings, serving regression |
| Grouped MoE | G128 scale generation and grouped tuning search corrected | Tune the actual EP16 shapes, inspect exported config, model regression |
| Communication | TP16 all-reduce/all-gather benchmark on each pair | Numerical check, slowest-rank latency, four-host NIC/RDMA counters |
| Long context | Existing resumable acceptance extended to selectable production profiles | Full 128K/512K/1M runs, not just successful allocation/startup |
| Quality | Kernel references and known-answer checks | Separate BF16 vs INT8 task evaluation still required; no lossless claim |
| Multimodal | Existing suite and tracing retained | Deferred large-image diagnosis, then all multimodal/mixed-load acceptance |

PIECEWISE still executes graph-break operations eagerly. Removing the indexer's
CPU length reads does not automatically make the entire model or NIXL transfer
graph-capturable. FULL Graph is not enabled by this change.

## Synchronize code

Run on P0. This only fast-forwards the existing INT8 branch; it does not switch
branches, discard changes, or stop running jobs. Stop the current managed serving
run with its actual run ID before starting these GPU tests.

```bash
bash <<'SH'
set -euo pipefail
cd /data/vllm
test "$(git branch --show-current)" = gfx1100/glm5next-int8
git pull --ff-only
for host in 10.5.10.3 10.5.10.55 10.5.10.56; do
  ssh -p 21985 -i ~/.ssh/rebond.pem "$host" \
    'cd /data/vllm && test "$(git branch --show-current)" = gfx1100/glm5next-int8 && git pull --ff-only && git log -1 --oneline'
done
.venv/bin/python -m pytest --version
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  gpu-check engines --run-id perf-adapt-a
SH
```

The supervisor checks every node's commit and available VRAM. Unknown GPU owners
are reported, not killed automatically. Existing `gpu-clean` accepts explicit
orphan worker PIDs after ownership inspection.

## Background A/B campaign

The five profiles isolate eager baseline, GPU indexer, Graph, and prefill batching.
Their order is `baseline`, `indexer`, `graph64`, `prefill2048`, `prefill8192`.
The last two differ from `graph64` only in the token budget per scheduler step.
The scheduler's maximum sequences is not reduced. Capture sizes up to 64 do not
impose a server concurrency limit.

The first pass uses 2K/32K prompts, output 256, concurrency 1/2/4/8/16/32.
The configured context remains 1M. This pass selects candidates; it is not full
production acceptance. Each profile runs functional checks before performance.

```bash
cd /data/vllm
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  start production --run-id perf-adapt-a
```

The command returns after detaching. The campaign first runs GPU indexer and
linear tests on P0 GPU0, including changed-input Graph replays. Skipped or empty
GPU tests fail the gate. It then starts fresh four-host engines and monitors per
profile, waits for health, runs the matrix, stops those engines and collects logs.
GPU compilation, numerical, startup or functional failure stops progression.

```bash
cd /data/vllm
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  status production --run-id perf-adapt-a
tail -n 100 /data/logs/glm53-assessment/perf-adapt-a/p0/production/console.log
cat /data/logs/glm53-assessment/perf-adapt-a/p0/production/data/progress.json
```

Stop using the campaign ID. This also stops the active child profile on all nodes,
even if the supervisor exceeded its shutdown grace period.

```bash
cd /data/vllm
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  stop production --run-id perf-adapt-a
```

Profile data lives under `/data/logs/glm53-assessment/perf-adapt-a-PROFILE`.
Generate a report for each completed or interrupted profile:

```bash
cd /data/vllm
for profile in baseline indexer graph64 prefill2048 prefill8192; do
  root=/data/logs/glm53-assessment/perf-adapt-a-$profile
  if [ -f "$root/p0/test/data/manifest.json" ]; then
    .venv/bin/python examples/disaggregated/glm5next_int8/assessment_report.py "$root"
  fi
done
```

Compare the same prompt length/concurrency: completed/failed requests, KV checks,
input/output throughput, TTFT and TPOT P95, per-node GPU/CPU utilization, VRAM,
NIC errors, and RDMA port counters. Do not infer bottlenecks from GPU busy alone
or add bond counters to physical-port counters. Monetary fields in the existing
report are optional estimates; they are not a qualification criterion.

## Communication microbenchmark

Run with engines and the performance campaign stopped. Both TP16 pairs are
measured independently; this measures RCCL collective latency, not NIXL transfer
bandwidth or an end-to-end PD request.

```bash
cd /data/vllm
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  start monitors --run-id comm-adapt-a
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  start communication --run-id comm-adapt-a
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  status communication --run-id comm-adapt-a
```

After completion (`exit_code=0` on all four nodes):

```bash
cd /data/vllm
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  stop all --run-id comm-adapt-a
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  collect all --run-id comm-adapt-a
cat /data/logs/glm53-assessment/comm-adapt-a/p0/collective/data/collectives.json
```

The D-pair result is in the collected D0 archive. NIXL transfer counters remain
recorded alongside every serving profile in its P/D Prometheus snapshots.

## Offline kernel tuning

Run with serving stopped. Supply linear `--shape N K` values from the actual TP
partitions; the example below benchmarks a 4096×4096 matrix only. Configs are
loaded only for an exact dtype/M/N/K match on the same device name and architecture.
Unmeasured shapes retain the existing launch parameters.

```bash
cd /data/vllm
mkdir -p /data/logs/glm53-tuning
nohup env HIP_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/kernels/benchmark_rocm_w8a16.py \
  --shape 4096 4096 --batch-sizes 1 2 4 8 16 32 64 128 512 2048 \
  --output /data/logs/glm53-tuning/linear.json \
  > /data/logs/glm53-tuning/linear.log 2>&1 < /dev/null &
echo $! > /data/logs/glm53-tuning/linear.pid
```

MoE tuning uses the checkpoint text config, TP16/EP16 and group size 128. Ray is
required by the existing MoE tuner. The output is a candidate, not automatically
installed into the serving runtime.

```bash
cd /data/vllm
mkdir -p /data/logs/glm53-tuning/moe
nohup env HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python \
  benchmarks/kernels/benchmark_moe.py \
  --model /data/models/zai-org/GLM-5.3-Flash-W8A16-G128 \
  --model-prefix text_config --tp-size 16 --enable-expert-parallel \
  --dtype int8_w8a16 --tune --batch-size 1 2 4 8 16 32 64 128 512 2048 \
  --save-dir /data/logs/glm53-tuning/moe \
  > /data/logs/glm53-tuning/moe.log 2>&1 < /dev/null &
echo $! > /data/logs/glm53-tuning/moe.pid
```

Run the two tuners sequentially. Review timings and correctness first, then copy
the same candidate files to all four nodes and set `VLLM_ROCM_W8A16_CONFIG` and
`VLLM_TUNED_CONFIG_FOLDER` in a new assessment config. Retest using a new run ID.
Changing a tuning file requires restarting workers; config loads are cached.

## Full text acceptance after choosing a profile

Use the measured winner, not an assumed winner. Example selecting `graph64`:

```bash
cd /data/vllm
mkdir -p /data/logs/glm53-assessment/perf-full-a
.venv/bin/python - <<'PY'
import json
from pathlib import Path

cfg = json.loads(Path('examples/disaggregated/glm5next_int8/assessment_config.json').read_text())
cfg['production_profiles'] = ['graph64']
cfg['production_cases'] = ['text2k', 'text32k', 'text128k', 'text512k', 'text1m']
Path('/data/logs/glm53-assessment/perf-full-a/config.json').write_text(json.dumps(cfg, indent=2))
PY
.venv/bin/python examples/disaggregated/glm5next_int8/assessment_jobs.py \
  start production --run-id perf-full-a \
  --config /data/logs/glm53-assessment/perf-full-a/config.json
```

Repeated-run stability, long-context retrieval, BF16/INT8 quality comparison and
the deferred multimodal gate must all be recorded before declaring production
acceptance complete. A text campaign completing does not close those gates.
