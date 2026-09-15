# GLM-5.3-Flash W8A16-G128: four-host ROCm PD test

This is a native Python deployment using each server's existing `/data/vllm`
checkout and `.venv`. Run commands manually on the named host. Each host has
eight GPUs. Two hosts form one TP16/EP prefiller, and two form one TP16/EP
decoder. This is two engines, not four independent TP8 replicas.

| Host | IP | Group | Node rank | API port |
| --- | --- | --- | --- | --- |
| P0 | 10.5.10.36 | Prefill TP16 | 0 | 8001 |
| P1 | 10.5.10.3 | Prefill TP16 | 1 | headless |
| D0 | 10.5.10.55 | Decode TP16 | 0 | 8002 |
| D1 | 10.5.10.56 | Decode TP16 | 1 | headless |

The initial profile retains image/video support and uses a 131072-token total
context limit, vLLM's default scheduler slot count, 512 scheduled tokens per step, chunked
prefill, and breakable PIECEWISE graphs up to 16 tokens. GPU utilization is 0.90:
TP16 has more weight headroom than the single-host TP8 test. Maximum concurrency
at full context depends on the measured KV capacity, not just scheduler slots.
These settings are a starting point for PD validation, not tuned performance.
The launcher does not impose a separate concurrency cap unless `MAX_NUM_SEQS`
is explicitly set. vLLM still schedules requests within its configured limits
and available KV capacity.

## Update and check all four hosts

Stop the previous test service in its terminal before starting a new engine on
the same GPUs. Stop an old BF16 PD test if it occupies these GPUs or ports.

On each host, use its existing checkout:

```bash
cd /data/vllm
git fetch origin gfx1100/glm5next-int8
git switch gfx1100/glm5next-int8
git pull --ff-only origin gfx1100/glm5next-int8
git log -1 --oneline
```

All four must have the same commit, compatible ROCm/PyTorch/native extensions,
and the same complete checkpoint at
`/data/models/zai-org/GLM-5.3-Flash-W8A16-G128`. Do not rebuild extensions for
Python-only changes. The check below verifies the ROCm GPU count, imports
`nixl_rocm._api`, prints the actual vLLM import path, and checks shard presence.
It does not verify file hashes, GPU-direct transport, or distributed inference.
If `nixl_rocm` cannot be imported, resolve the ROCm NIXL environment first;
installing the CUDA `nixl` dependency is not a substitute.

Run only the corresponding line on each host:

```bash
# P0
bash examples/disaggregated/glm5next_int8/launch_pd.sh p0 --check
# P1
bash examples/disaggregated/glm5next_int8/launch_pd.sh p1 --check
# D0
bash examples/disaggregated/glm5next_int8/launch_pd.sh d0 --check
# D1
bash examples/disaggregated/glm5next_int8/launch_pd.sh d1 --check
```

The launcher detects the interface owning the configured IP. `IFACE_NAME` can
override it. `P0_IP`, `P1_IP`, `D0_IP`, and `D1_IP` can override addresses; use
the same address map on all hosts. Existing network policy must permit the
group rendezvous, distributed worker connections, NIXL side channels, and UCX
transport. Rendezvous ports are 29501/29502; NIXL side-channel bases are
5557/5657. Merely opening the API ports is insufficient.

`--dry-run` prints the complete command without loading the model. The script
uses `VLLM_SSM_CONV_STATE_LAYOUT=DS` for KDA transfer, and `LBHNC` on both sides.
It preserves the configured UCX transport rather than forcing TCP or claiming
RDMA is available. Transfer failures use the connector's `fail` policy.

## Start the engines

The launcher uses `python -m vllm.entrypoints.cli.main serve` so that
`--headless` dispatches follower nodes to the headless multiprocess executor.
The legacy `python -m vllm.entrypoints.openai.api_server` entrypoint bypasses
that dispatch and causes `collective_rpc should not be called on follower node`
on P1/D1.

Run each host's command in a separate persistent terminal. Start P0 and then P1
without waiting for P0 health; do the same for D0 and D1. Each leader waits for
its peer to join.

```bash
# P0
bash examples/disaggregated/glm5next_int8/launch_pd.sh p0
# P1
bash examples/disaggregated/glm5next_int8/launch_pd.sh p1
# D0
bash examples/disaggregated/glm5next_int8/launch_pd.sh d0
# D1
bash examples/disaggregated/glm5next_int8/launch_pd.sh d1
```

Logs are saved per host at `/data/logs/glm53-int8-pd-ROLE/latest.log`.
`MODEL_DIR`, `LOG_DIR`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_BATCHED_TOKENS`, and
`GPU_MEMORY_UTILIZATION` are launcher overrides. Keep model/context/cache
configuration consistent between groups for this first test.

For a 1048576-token context, prefix each host's launch command with
`MAX_MODEL_LEN=1048576`. Run `unset MAX_NUM_SEQS` first if an earlier test
exported it; otherwise that explicit concurrency override remains active.
The limit includes both input and output tokens. Validate the reported KV
capacity and long-request transfer on both groups before measuring performance.

From another P0 terminal, wait until both return HTTP 200:

```bash
curl --noproxy '*' -i --max-time 10 http://10.5.10.36:8001/health
curl --noproxy '*' -i --max-time 10 http://10.5.10.55:8002/health
```

P1 and D1 do not expose API listeners.

## Start the test proxy on P0

The repository's NIXL integration proxy submits prefill to P0 and forwards the
returned transfer metadata to D0. This proxy is a test utility, not a production
gateway. Its health route is `/healthcheck`.

```bash
cd /data/vllm
mkdir -p /data/logs/glm53-int8-pd-proxy
set -o pipefail
NO_PROXY=127.0.0.1,localhost,10.5.10.36,10.5.10.55 \
no_proxy=127.0.0.1,localhost,10.5.10.36,10.5.10.55 \
.venv/bin/python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --host 127.0.0.1 --port 8000 \
  --prefiller-hosts 10.5.10.36 --prefiller-ports 8001 \
  --decoder-hosts 10.5.10.55 --decoder-ports 8002 \
  2>&1 | tee /data/logs/glm53-int8-pd-proxy/startup.log
```

If overriding host addresses, update the proxy and validation commands too.

## Validate generation and actual transfer

From another P0 terminal, save the decoder metrics before and after a request:

```bash
curl --noproxy '*' -fsS http://10.5.10.55:8002/metrics \
  > /data/logs/glm53-int8-pd-proxy/metrics-before.txt

curl --noproxy '*' --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-int8","messages":[{"role":"user","content":"请用一句话解释什么是张量并行。"}],"temperature":0,"max_tokens":256,"stream":false}' \
  -w '\nTotal request time: %{time_total} seconds\n'

curl --noproxy '*' -fsS http://10.5.10.55:8002/metrics \
  > /data/logs/glm53-int8-pd-proxy/metrics-after.txt

grep -E '^vllm:nixl_(bytes_transferred_(sum|count)|num_failed_(transfers|notifications)_total)' \
  /data/logs/glm53-int8-pd-proxy/metrics-before.txt \
  /data/logs/glm53-int8-pd-proxy/metrics-after.txt
```

Require a sensible response, an increase in transfer bytes/count, and no
increase in failed transfers/notifications. API health alone does not prove PD
transfer worked. If metrics have not been published yet, collect the after
snapshot again after the next metrics update.

After that, validate 2K, 32K, and near-128K requests through port 8000, including
prompt lengths not divisible by four to exercise the transferred kpool tail.
Compare deterministic responses with direct inference before treating the PD
configuration as accepted. Test image/video requests separately. For capacity
and revenue comparisons, charge this deployment for 32 GPUs and include
end-to-end proxy latency and transfer time.

## Validation status

The launcher is checked locally with Bash syntax validation and four role
dry-runs. GPU startup, ROCm NIXL transport, model outputs, and PD performance
must be validated on the four hosts. Single-host INT8 inference success is not
a substitute for these checks.

## Functional acceptance and performance

Run the committed client on P0, against the PD proxy on port 8000. All four
servers must use `MAX_MODEL_LEN=1048576`. Unset `MAX_NUM_SEQS` to leave scheduler
concurrency at the vLLM default. The client's concurrency sweep below specifies
offered load; it does not change server limits.

The client requires `httpx`, `Pillow`, `pybase64`, `regex`, and `transformers` in the existing
`.venv`, the local model tokenizer, and `ffmpeg` on PATH with a working `libx264`
encoder. It generates its own 448x448 images and a nine-second MP4 in the result
directory and sends media inline. No public media download or cross-host media
mount is needed. Keep the proxy's `NO_PROXY` settings from the command above.

```bash
cd /data/vllm
.venv/bin/python -c 'import httpx, PIL, pybase64, regex, transformers'
ffmpeg -version
export PD_RESULTS=/data/logs/glm53-int8-pd-tests/$(date +%Y%m%d-%H%M%S)
mkdir -p "$PD_RESULTS"
set -o pipefail

.venv/bin/python examples/disaggregated/glm5next_int8/run_checks.py functional \
  --output-dir "$PD_RESULTS" \
  2>&1 | tee "$PD_RESULTS/functional.log"
```

Functional cases run in this order and stop at the first failure:

| Case | Input | Acceptance |
| --- | --- | --- |
| text2k | 2047 tokens | Retrieve all three random secrets |
| text32k | 32767 tokens | Retrieve secrets near start, middle, end |
| text128k | 131071 tokens | Same, with exact server token-count check |
| text512k | 524287 tokens | Same |
| text1m | 1047551 tokens | Same, reserving 1024 output tokens below 1048576 |
| image | Red image | Correct color JSON |
| images | Red and blue images | Correct image order JSON |
| video_frames | Three JPEG frames via video/jpeg | Red, green, blue in time order |
| video_mp4 | Nine-second MP4 | Same order, repeated colors collapsed |
| mixed | Blue image and MP4 in one request | Correct image and video answers |

Text uses the checkpoint chat template and sends exact token IDs to
`/v1/completions`. Media uses `/v1/chat/completions`. Prompt lengths deliberately
leave a partial kpool group. Every request gets a unique prefix to avoid
reusing the long text prefix cache across test requests. These synthetic
retrieval/color cases verify the execution path; they are not a general model
quality benchmark or a combined 1M-text-plus-video capacity test.

Both text and media requests use `reasoning_effort="low"`. The checkpoint
template ignores `enable_thinking`, defaults to Max effort, and always opens
`<think>` for generation. Low effort still allows reasoning, which consumes
the output token budget; it does not disable thinking.

Each functional case requires a complete streaming response, token usage,
the expected answer, increasing decoder NIXL byte/count metrics, and unchanged
transfer/notification failure counters. Raw P/D metrics are saved before and
after every stage, including available KV-cache and preemption metrics. Metric
publication is polled for up to 20 seconds. Missing metrics fail acceptance.
Keep unrelated traffic off these engines during acceptance so metric deltas
can be attributed to the test. The request timeout defaults to four hours;
`--timeout` changes it. A pending request does not print incremental progress.

### Performance after acceptance

`perf` requires a complete, successful `functional.json` in the same result
directory with matching client commit, model, tokenizer path and endpoints.
Rerun functional acceptance after changing the deployed model, server code or
cache configuration; the client cannot independently attest remote processes.

First sweep short text and heterogeneous text/media traffic:

```bash
.venv/bin/python examples/disaggregated/glm5next_int8/run_checks.py perf \
  --output-dir "$PD_RESULTS" \
  --cases text2k,text32k,mixed_load \
  --concurrency 1 2 4 8 16 32 --requests 32 \
  2>&1 | tee "$PD_RESULTS/perf-short-mixed.log"
```

Then sweep long context:

```bash
.venv/bin/python examples/disaggregated/glm5next_int8/run_checks.py perf \
  --output-dir "$PD_RESULTS" \
  --cases text128k,text512k,text1m \
  --concurrency 1 2 4 8 16 32 --requests 16 \
  2>&1 | tee "$PD_RESULTS/perf-long.log"
```

Test each media mode independently if needed:

```bash
.venv/bin/python examples/disaggregated/glm5next_int8/run_checks.py perf \
  --output-dir "$PD_RESULTS" \
  --cases image,images,video_frames,video_mp4,mixed \
  --concurrency 1 2 4 8 16 32 --requests 32 \
  2>&1 | tee "$PD_RESULTS/perf-media.log"
```

`mixed_load` cycles 2K text, 32K text, single image, two images, MP4, and an
image-plus-video request. Each level sends at least the larger of `--requests`
and four times concurrency, using a closed-loop client with that many requests
in flight. Requests waiting for a client slot are excluded from per-request
latency; the full stage duration is used for aggregate throughput. Longer
sustained runs can increase `--requests`; levels above 32 can be passed directly.
Large-context, high-concurrency levels can take many hours and substantial
client RAM because distinct prompts are prepared before the timed interval.
Performance media also gets a unique small corner pattern per request, changing
the decoded pixels to avoid measuring repeated-media processor-cache hits.
Media encoding is outside the timed interval; fixtures remain under
`assets/STAGE/REQUEST_INDEX` for inspection.
The client stops the sweep on request or transfer failure, preserving results.

Performance requests use `ignore_eos` and a fixed 256 output tokens (override
with `--output-tokens`, up to 1024). Performance measures successful transport
and token production; semantic correctness is checked in the functional run.
There are no extra warmup requests; the first level includes any remaining
cold compilation and should be reported separately from repeated steady runs.

Outputs include per-request `*.jsonl`, per-stage `*-summary.json`, raw `*.prom`,
`functional.json`, and `performance.json`. The last file describes the latest
sweep; timestamped stage files from earlier sweeps remain intact. Reports give
successful input/output tokens per second, requests per second, failures, and
P50/P95/P99 TTFT, TPOT and end-to-end latency in seconds. TPOT is the interval
from first to last output-bearing stream event divided by output tokens minus
one; it is not a per-token ITL distribution. One-token responses have no TPOT.
Failed requests do not contribute to token throughput. Proxy, transfer and
server queue time are included; monitor client CPU/network utilization too so
a client bottleneck is not mistaken for server capacity.

Server metrics do not replace hardware measurements. Record `rocm-smi` power,
utilization and memory on all four hosts during sustained runs. Revenue uses
successful aggregate throughput, actual input/output prices and measured power
for all 32 GPUs. This client provides acceptance and load-test evidence; it
does not turn the toy proxy into a production gateway or verify failover.

CPU-only client regression checks (no server or checkpoint required):

```bash
.venv/bin/python -m pytest --confcutdir=tests/benchmarks \
  tests/benchmarks/test_glm53_pd_checks.py -q
```

## UCX registration diagnostics and remaining PD acceptance

### Repeated output or non-finite logprobs

For a failure at `stage=gemm2 output`, add
`VLLM_GLM5NEXT_DUMP_DIR=/data/logs/glm53-int8-gemm2` to both D0/D1 launch
environments alongside the eager finite checks below. On a failing W8A16
second GEMM, each affected worker saves its actual operands and alignment
metadata before the existing finite guard aborts. Finite outputs create no
files. Snapshots contain local expert weights and request activations; keep
them on the server. This is opt-in diagnostic code, not a numerical fix.

After the failed engine's workers have exited, replay one snapshot on the same
host (replace the filename with the `GLM GEMM2 capture:` path from its log):

```bash
/data/vllm/.venv/bin/python /data/vllm/vllm/model_executor/layers/fused_moe/wna16_debug.py /data/logs/glm53-int8-gemm2/gemm2-rank0-pidXXXX-XXXXXXXX.pt --device cuda
```

This computes a CPU reference and runs the isolated kernel three times on GPU 0,
without loading the model, starting distributed workers, or using NIXL. To test
another physical GPU, set `HIP_VISIBLE_DEVICES=4` before the command. Omit
`--device cuda` for CPU reference only. `--block-size-k 32` (or 64/128) allows
a controlled tile-size comparison. The report includes original/replay
non-finite counts, reference results, operand ranges and strides. A finite
reference with a failing replay implicates the isolated kernel path; a passing
replay leaves live workspace/lifetime interactions to investigate. Finite
output alone is not an accuracy pass: inspect numerical differences too.

Use `--workspace-mib N` with `--device cuda` to replay A and C as disjoint
views of one N-MiB allocation, preserving strides and the modular MoE buffer
offset. The default uses independent allocations. Compare 1, 1321 and 2641
MiB: with a 128-dimensional FP8 index head, the indexer reserves
`40 * max_model_len * 132 + 1 MiB`, giving 1321 MiB at 256K and 2641 MiB at
512K. These are indexer reservations, not measurements of the final workspace
after other users may grow it. This experiment tests sharing and allocation
size; it does not reproduce the full model's preceding writes or stream timing.

Add `--audit-workspace` to compare every restored tensor's bytes with the
snapshot, then run a small Triton probe before GEMM2. It reports the A/C
addresses observed inside the GPU kernel, copies A into an independent buffer,
and fills C with 3. A mismatched address is reported without dereferencing it.
Each subsequent GEMM2 replay also checks A and the NaN output sentinel before
launch. This distinguishes restoration, pointer conversion, basic kernel
access and GEMM2 failures. The audit adds synchronization and allocations;
compare with ordinary replay if the failure disappears. No serving path changes.

The ROCm INT8 WNA16 launcher now automatically exposes A/C view bounds when
their backing storage exceeds `2**31 - 1` bytes but their accessed spans do not.
This applies to GEMM1 and GEMM2 and preserves allocation, addresses, strides
and reuse. Triton 3.7.1 otherwise drops `tt.pointer_range = 32` based on the
backing storage size. A captured GLM GEMM2 failed three times in a 2641 MiB
allocation and matched the independent-allocation reference error three times
when given view bounds instead. CUDA, INT4, small backing storage and views
that themselves exceed the bound retain their previous dispatch.

Default GPU replay now exercises this launcher fix: no `--view-pointer-range`
or isolation flag is needed. The former option remains for explicit compiler
experiments, using the same TensorView helper. Keep
`VLLM_GLM5NEXT_ISOLATE_GEMM2` unset when validating the fix in the full model.
Compare numerical error, not just NaNs. The GPU regression covering both
GEMMs and remote-expert zero output is:

```bash
.venv/bin/python -m pytest tests/kernels/moe/test_moe.py -k w8a16_shared_workspace_pointer_range -q
```

The earlier `VLLM_GLM5NEXT_ISOLATE_GEMM2=1` diagnostic remains available,
but did not correct the observed 512K model output. It requires
`VLLM_GLM5NEXT_CHECK_FINITE=1` eager diagnostics.
It allocates only the INT8 GEMM2 output independently instead of reusing the
GEMM1 workspace; inputs, weights, launch configuration and reduction stay the
same. The default is off. This is a diagnostic comparison, not a confirmed
fix. Additional live storage is `tokens * top_k * hidden_size * dtype_size`
bytes (128 KiB for the observed two-token BF16 case), plus allocator overhead.

A completed HTTP request with repetitive output is not a correctness pass.
Compare a fresh prompt sent directly to D0 with a request through the proxy;
reusing a prompt can reuse prefix state from an earlier request. A JSON error
reporting `nan` in logprobs requires numerical investigation, not a longer RPC
timeout or a change to sampling penalties.

For a controlled graph-versus-eager comparison, restart both D0 and D1 with
`ENFORCE_EAGER=1` added to their existing launch environments. Preserve the
working UCX settings and `MAX_MODEL_LEN=1048576`. This selects `--enforce-eager`
and disables breakable graphs; it does not change concurrency, TP/EP, cache
dtype or async scheduling. The default remains `ENFORCE_EAGER=0` (piecewise
graphs). Apply the same setting on both nodes of a TP group.

Also set `VLLM_RAISE_ON_LOGIT_NANS=1` during diagnosis. This existing vLLM switch
counts NaNs in raw logits and raises with affected request IDs instead of
continuing to sample. It can stop the diagnostic engine on a bad request.
To locate an earlier failing module, additionally enable
`VLLM_GLM5NEXT_CHECK_FINITE=1` on both nodes of the diagnostic TP group. This
requires `--enforce-eager` and installs checks at decoder, attention/MLP,
linear, embedding and normalization boundaries. It reports the first observed
non-finite input, output or directly owned floating-point parameter with the
module path, TP rank, shape, dtype and count. Parameters are checked on first
use; buffers and forwards without attention metadata are skipped. It does not
inspect every internal fused operation or prove cache correctness. A module's
output failure narrows the boundary; it does not by itself identify the kernel
or rank that originally produced bad data before a collective.
MoERunner diagnostics also check routed parameters and bracket expert selection,
modular expert execution, routed scaling and the runner's reduction methods.
For Triton WNA16 experts it additionally checks GEMM1, activation, GEMM2 and
local top-k summation, without inspecting unwritten output workspaces.
An `input` failure at a reduction means bad data arrived before that call;
an `output` failure with finite input needs results from all ranks to distinguish
another rank's bad contribution, overflow and a communication implementation bug.
The new runner's scheduler-based kernel warmup has attention metadata and is
checked too. It uses synthetic token IDs and real prefill/decode execution:
a failure there must be labeled as warmup, not an actual client request.
The checks synchronize GPU work and may change timing, so use short fresh
prompts for diagnosis, not the performance suite. The switch defaults off and
does not replace or sanitize values. Disable it before performance testing.
NaN logprobs alone do not prove raw logits contain NaNs: infinities can also
make log-softmax non-finite. Retain the request/logprobs and worker traceback.
Successful eager output would narrow the investigation to graph-related
execution; it would not certify the original graph configuration or PD state
transfer. Restore graph mode only after resolving and retesting the cause.

### Registration checks

The launcher now registers and deregisters an 8 MiB host buffer and an 8 MiB
VRAM buffer on each GPU before loading model weights. A failed probe stops
startup. `NIXL_PREFLIGHT_MIB` changes this probe size; it does not change model
context, KV-cache capacity or concurrency. Small-buffer success cannot prove
that a full KV allocation or simultaneous eight-worker registration succeeds.

For `Failed to ucp_mem_map` / `NIXL_ERR_BACKEND`, retain the earlier native UCX
error. For example, `ibv_reg_mr ... Invalid argument` on `mlx5_bond_0` while
registering `(rocm)` memory identifies a GPU memory registration failure in the
RDMA path. It does not establish a model OOM or a Kpool descriptor error.
`NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` do not select UCX network devices.

Run this on the affected host after stopping its failed engine and releasing
its workers. It loads no model and makes no connection to the other hosts:

```bash
cd /data/vllm
set -o pipefail
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python \
  examples/disaggregated/glm5next_int8/check_nixl.py \
  --sizes-mib 8 2048 16934 \
  2>&1 | tee /data/logs/glm53-int8-nixl-registration.log
```

The largest probe approximates the reported 17,756,385,280-byte KV allocation.
Buffers are allocated one at a time in fresh subprocesses. The report includes
host versus GPU results, GPU index, size, memlock limits, allocator/UCX settings,
and the UCX libraries actually loaded by NIXL. `ucx_info` on PATH may describe a
different UCX installation from the wheel. Exit status is nonzero if any case
fails or exceeds its timeout. No automatic transport fallback is performed.

Interpretation:

- Host succeeds, even small VRAM fails: investigate the ROCm peer-memory/DMA-BUF
  path, NIC driver and the loaded UCX build before retrying model startup.
- Small VRAM succeeds but large VRAM fails: inspect native UCX errors for
  registration-size/resource constraints or allocator behavior. This probe
  identifies the boundary; it does not repair the driver.
- All local probes pass: test full engine registration next, then inter-host
  handshakes and real payload transfer. Local registration is not an RDMA
  bandwidth, reachability or correctness test.

For an explicit TCP comparison only, repeat the 8 MiB probe with
`UCX_TLS=tcp,sm,self,rocm UCX_NET_DEVICES=all` in the environment. ROCm transports
must remain enabled to recognize/copy GPU buffers (see the
[UCX transport documentation](https://openucx.readthedocs.io/en/master/faq.html#which-transports-does-ucx-use)).
Passing this comparison does not certify the RDMA path. Record the transport
in performance results; do not substitute TCP throughput for RDMA capacity.

The remaining deployment acceptance items are:

- Both TP16 groups use identical code/model and compatible transfer geometry.
  P0/P1 and D0/D1 are two distributed engines, not four independent replicas.
- Scheduler side channels are on P0:5557 and D0:5657. The NIXL/UCX data path
  additionally needs cross-host connectivity; opening HTTP ports alone is
  insufficient. Keep handshake compatibility checking enabled.
- The toy proxy's health endpoint only checks its own process. It can forward
  to decode without transfer metadata if prefill returns none. Continue to
  require actual decoder NIXL byte/count increases and no failures in
  `run_checks.py`; a successful HTTP answer alone is not PD acceptance.
- Prefix reuse, non-block-aligned prompts, long-context output correctness,
  multimodal transfer and request cancellation/restart still need four-host
  execution. The included functional suite covers retrieval and media content;
  it does not yet certify cancellation cleanup, failover or long-duration soak.
- Performance starts after functional acceptance, measuring all 32 GPUs and
  the selected transport. Local CPU tests and registration probes do not prove
  production readiness.

## Background full assessment

See [the P0-managed full assessment guide](ASSESSMENT.md) for detached shell-session
jobs, per-stage resume, 1M and multimodal tests, four-host monitoring and reports.
It does not use systemd or create server worktrees.
