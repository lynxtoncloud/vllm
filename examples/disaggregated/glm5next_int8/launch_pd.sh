#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

role=${1:?Usage: launch_pd.sh p0|p1|d0|d1 [--check|--dry-run]}
mode=${2:-run}
case "$mode" in run|--check|--dry-run) ;; *) exit 2 ;; esac

p0=${P0_IP:-10.5.10.36}
p1=${P1_IP:-10.5.10.3}
d0=${D0_IP:-10.5.10.55}
d1=${D1_IP:-10.5.10.56}
case "$role" in
  p0) node_ip=$p0; master=$p0; rank=0; port=8001; master_port=29501; side_port=5557; kv_role=kv_producer ;;
  p1) node_ip=$p1; master=$p0; rank=1; port=8001; master_port=29501; side_port=5557; kv_role=kv_producer ;;
  d0) node_ip=$d0; master=$d0; rank=0; port=8002; master_port=29502; side_port=5657; kv_role=kv_consumer ;;
  d1) node_ip=$d1; master=$d0; rank=1; port=8002; master_port=29502; side_port=5657; kv_role=kv_consumer ;;
  *) echo "Unknown role: $role" >&2; exit 2 ;;
esac

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_dir"
unset PYTHONPATH VLLM_ROCM_GFX1100_GLM53
model_dir=${MODEL_DIR:-/data/models/zai-org/GLM-5.3-Flash-W8A16-G128}
log_dir=${LOG_DIR:-/data/logs/glm53-int8-pd-$role}
iface=${IFACE_NAME:-}
if [[ -z "$iface" ]]; then
  iface=$(ip -o -4 addr show | awk -v target="$node_ip" \
    'split($4,a,"/") && a[1]==target {sub(/@.*/,"",$2); print $2; exit}')
fi
: "${iface:?Set IFACE_NAME to the interface owning the node IP}"

cmd=(env
  'HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7'
  VLLM_WORKER_MULTIPROC_METHOD=spawn
  VLLM_USE_BREAKABLE_CUDAGRAPH=1
  VLLM_ENGINE_READY_TIMEOUT_S=3600
  "VLLM_HOST_IP=$node_ip"
  "VLLM_NIXL_SIDE_CHANNEL_HOST=$node_ip"
  "VLLM_NIXL_SIDE_CHANNEL_PORT=$side_port"
  "GLOO_SOCKET_IFNAME=$iface" "NCCL_SOCKET_IFNAME=$iface"
  VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_KV_CACHE_LAYOUT=LBHNC
  .venv/bin/python -m vllm.entrypoints.openai.api_server
  --model "$model_dir" --served-model-name glm53-int8
  --host "$node_ip" --port "$port"
  --distributed-executor-backend mp
  --tensor-parallel-size 16 --enable-expert-parallel
  --nnodes 2 --node-rank "$rank"
  --master-addr "$master" --master-port "$master_port"
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$kv_role\",\"kv_load_failure_policy\":\"fail\"}"
  --no-disable-hybrid-kv-cache-manager
  --quantization compressed-tensors --dtype bfloat16 --kv-cache-dtype auto
  --max-model-len "${MAX_MODEL_LEN:-131072}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-512}"
  --enable-chunked-prefill
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --compilation-config '{"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":16}')
if [[ "$rank" == 1 ]]; then
  cmd+=(--headless)
fi
if [[ "$mode" == --dry-run ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

ip -o -4 addr show dev "$iface" | awk -v target="$node_ip" \
  'split($4,a,"/") && a[1]==target {found=1} END {exit !found}' || {
  echo "$iface does not own $node_ip" >&2; exit 1;
}
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/python - "$model_dir" <<'PY'
import importlib
import json
import sys
from pathlib import Path

import torch
import vllm

if not torch.version.hip or torch.cuda.device_count() != 8:
    raise RuntimeError("Each node must expose 8 ROCm GPUs")
nixl_api = importlib.import_module("nixl_rocm._api")
for attr in ("nixl_agent", "nixl_agent_config"):
    if not callable(getattr(nixl_api, attr, None)):
        raise RuntimeError(f"nixl_rocm._api lacks {attr}")
path = Path(sys.argv[1])
json.loads((path / "config.json").read_text())
index = json.loads((path / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
missing = sorted(name for name in shards if not (path / name).is_file())
if missing:
    raise RuntimeError(f"Missing model shards: {missing}")
print("vLLM:", vllm.__file__)
print("PyTorch:", torch.__version__, "ROCm:", torch.version.hip)
print("NIXL ROCm:", nixl_api.__file__)
print("Model shards present:", len(shards))
PY
git log -1 --oneline
[[ "$mode" == --check ]] && exit 0

mkdir -p "$log_dir"
log_file="$log_dir/$(date +%Y%m%d-%H%M%S)-$$.log"
ln -sfn "$(basename "$log_file")" "$log_dir/latest.log"
"${cmd[@]}" 2>&1 | tee "$log_file"
