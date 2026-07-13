#!/usr/bin/env bash
# Stage 2: dataset condensation (embed + influence + probe_cos, scale x100).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Config can be overridden per machine, either positionally or via env:
#   bash run_condense.sh ../cifar10_ipc50.yaml
#   CFG=../cifar10_ipc50.yaml bash run_condense.sh
CFG="${1:-${CFG:-../cifar10_ipc50.yaml}}"
# Per-config log so the 4 fusion variants don't overwrite each other's logs.
LOG="condense_$(basename "${CFG%.*}").log"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "[Stage2] config=$CFG gpus=$GPUS log=$LOG start=$(date '+%F %T')"

torchrun --nproc_per_node="$NPROC" --nnodes=1 --master_port="$PORT" condense.py \
  --gpu="$GPUS" --ipc=50 \
  --run_mode=Condense \
  --config_path="$CFG" 2>&1 | tee "$LOG"

echo "[Stage2] FINISHED end=$(date '+%F %T')"
