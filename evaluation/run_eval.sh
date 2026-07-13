#!/usr/bin/env bash
# Stage 3: evaluate condensed dataset (10 repeats by default).
# Usage:
#   bash run_eval.sh
#   bash run_eval.sh /abs/path/to/data_20000.pt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CFG="../cifar10_ipc50.yaml"
TAG="cifar10_ipc50"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
LOG="${LOG:-eval_cifar10_ipc50.log}"

if [ -n "${1:-}" ]; then
  LOAD="$1"
else
  RUN=$(ls -td ../results/condense/cifar10/ipc50/${TAG}_adamw_lr_img_0.0010_numr_reqs4096_factor2_* 2>/dev/null | head -1)
  if [ -z "$RUN" ]; then
    echo "[Stage3] ERROR: no condense run dir found for TAG=$TAG"
    exit 2
  fi
  LOAD="$RUN/distilled_data/data_20000.pt"
  if [ ! -f "$LOAD" ]; then
    LOAD=$(ls -t "$RUN"/distilled_data/data_*.pt 2>/dev/null | grep -v data_init | head -1)
  fi
fi

if [ ! -f "$LOAD" ]; then
  echo "[Stage3] ERROR: load_path not found: $LOAD"
  exit 2
fi

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')
echo "[Stage3] load=$LOAD start=$(date '+%F %T')"

torchrun --nproc_per_node="$NPROC" --nnodes=1 --master_port="$PORT" evaluation.py \
  --gpu="$GPUS" --ipc=50 \
  --run_mode=Evaluation \
  --config_path="$CFG" \
  --load_path="$LOAD" 2>&1 | tee "$LOG"

echo "[Stage3] FINISHED end=$(date '+%F %T')"
grep "Mean Accuracy" "$LOG" | tail -1 || true
