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
