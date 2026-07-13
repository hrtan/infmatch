#!/usr/bin/env bash
# Stage 1: trajectory pretraining (20 convnets, save epoch ckpts every 2 epochs).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CFG="../cifar10_ipc50.yaml"
LOG="pretrain_cifar10_ipc50.log"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "[Stage1] config=$CFG gpus=$GPUS start=$(date '+%F %T')"

torchrun --nproc_per_node="$NPROC" --nnodes=1 --master_port="$PORT" pretrain.py \
  --gpu="$GPUS" --ipc=50 \
  --run_mode=Pretrain \
  --config_path="$CFG" 2>&1 | tee "$LOG"

echo "[Stage1] FINISHED end=$(date '+%F %T')"
