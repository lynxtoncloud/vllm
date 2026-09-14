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
