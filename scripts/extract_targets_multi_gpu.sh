#!/usr/bin/env bash
set -euo pipefail

split="${1:-train}"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a gpus <<< "${gpu_list}"
num_shards="${#gpus[@]}"
mkdir -p logs

for shard_id in "${!gpus[@]}"; do
  gpu="${gpus[$shard_id]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/extract_targets.py \
    --split "${split}" --device cuda:0 \
    --num-shards "${num_shards}" --shard-id "${shard_id}" \
    > "logs/extract_${split}_${shard_id}.log" 2>&1 &
  pids[$shard_id]=$!
  echo "shard ${shard_id}/${num_shards} -> GPU ${gpu}, pid ${pids[$shard_id]}"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
exit "${status}"
