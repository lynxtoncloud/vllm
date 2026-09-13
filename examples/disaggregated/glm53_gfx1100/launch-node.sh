#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

role=${1:?Usage: launch-node.sh p0|p1|d0|d1 [--dry-run]}
case "$role" in
  p0) node_ip=10.5.10.36; master=10.5.10.36; rank=0; port=8001; master_port=29501; side_port=5557; kv_role=kv_producer ;;
  p1) node_ip=10.5.10.3; master=10.5.10.36; rank=1; port=8001; master_port=29501; side_port=5557; kv_role=kv_producer ;;
  d0) node_ip=10.5.10.55; master=10.5.10.55; rank=0; port=8002; master_port=29502; side_port=5657; kv_role=kv_consumer ;;
  d1) node_ip=10.5.10.56; master=10.5.10.55; rank=1; port=8002; master_port=29502; side_port=5657; kv_role=kv_consumer ;;
  *) echo "Unknown role: $role" >&2; exit 2 ;;
esac
dry_run=${2:-}
[[ -z "$dry_run" || "$dry_run" == --dry-run ]] || exit 2
image=${IMAGE:-local/vllm:glm53-gfx1100}
model_dir=${MODEL_DIR:-/data/models/zai-org/GLM-5.3-Flash-BF16}
name=${CONTAINER_NAME:-glm53-src-$role}
log_dir=${LOG_DIR:-/data/logs/glm53-src-$role}
hf_cache_dir=${HF_CACHE_DIR:-/data/cache/huggingface}
iface=${IFACE_NAME:-}
if [[ -z "$iface" ]]; then
  iface=$(ip -o -4 addr show | awk -v target="$node_ip" 'split($4,a,"/") && a[1]==target {sub(/@.*/,"",$2); print $2; exit}')
fi
: "${iface:?Set IFACE_NAME to the interface owning the node IP}"
cmd=(docker run --init --name "$name" --network host --ipc host
  --ulimit memlock=-1:-1)
if [[ -d /dev/infiniband ]]; then
  cmd+=(--device /dev/infiniband)
fi
cmd+=(
  --device /dev/kfd --device /dev/dri --group-add video
  --security-opt seccomp=unconfined
  --mount "type=bind,src=$model_dir,dst=/models/GLM-5.3-Flash-BF16,readonly"
  --mount "type=bind,src=$log_dir,dst=/app/logs"
  --mount "type=bind,src=$hf_cache_dir,dst=/root/.cache/huggingface"
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  -e VLLM_ROCM_GFX1100_GLM53=1
  -e VLLM_ROCM_USE_AITER=0 -e VLLM_ROCM_USE_AITER_MOE=0
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e "VLLM_HOST_IP=$node_ip"
  -e "VLLM_NIXL_SIDE_CHANNEL_HOST=$node_ip"
  -e "VLLM_NIXL_SIDE_CHANNEL_PORT=$side_port"
  -e "GLOO_SOCKET_IFNAME=$iface" -e "NCCL_SOCKET_IFNAME=$iface"
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e VLLM_KV_CACHE_LAYOUT=HND
  -e NCCL_DEBUG=WARN -e PYTHONFAULTHANDLER=1
  "$image" /models/GLM-5.3-Flash-BF16
  --served-model-name zai-org/GLM-5.3-Flash-BF16
  --host "$node_ip" --port "$port"
  --tensor-parallel-size 16 --enable-expert-parallel
  --nnodes 2 --node-rank "$rank" --master-addr "$master" --master-port "$master_port"
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$kv_role\",\"kv_load_failure_policy\":\"fail\"}"
  --no-disable-hybrid-kv-cache-manager
  --dtype bfloat16 --kv-cache-dtype auto --enforce-eager
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2048
  --gpu-memory-utilization 0.90
  --attention-backend ROCM_AITER_MLA_SPARSE
  --kernel-config '{"moe_backend":"triton","ir_op_priority":{"rms_norm":["native"],"fused_add_rms_norm":["native"]}}'
  --compilation-config '{"cudagraph_mm_encoder":false}'
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45)
if [[ "$rank" == 1 ]]; then
  cmd+=(--headless)
fi
if [[ "$dry_run" == --dry-run ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi
[[ -f "$model_dir/config.json" ]] || { echo "Missing $model_dir/config.json" >&2; exit 1; }
ip -o -4 addr show dev "$iface" | awk -v target="$node_ip" '
  split($4,a,"/") && a[1]==target {found=1} END {exit !found}' || {
  echo "$iface does not own $node_ip" >&2; exit 1;
}
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "Container $name already exists; inspect/stop it before a new run." >&2
  exit 1
fi
mkdir -p "$log_dir" "$hf_cache_dir"
log_file="$log_dir/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
ln -sfn "$(basename "$log_file")" "$log_dir/latest.log"
"${cmd[@]}" 2>&1 | tee "$log_file"
