#!/usr/bin/env bash
# One-command reproduction of every result in the revised manuscript.
# Usage:  bash revision/run_all.sh [OUT_DIR]      (default: outputs)
# Two lanes run in parallel, one per GPU (on a single-GPU machine both lanes share it).
# Safe to re-run: finished runs are skipped; runs are only STARTED before the deadline.
set -uo pipefail
OUT=${1:-outputs}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export PIPELINE_T0=${PIPELINE_T0:-$(date +%s)}
export PYTHONUNBUFFERED=1
LOG="$OUT/pipeline_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo "### start $(date -u)"
python -c "import torch, transformers, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"
NGPU=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
G1=$(( NGPU > 1 ? 1 : 0 ))

DATA="$OUT/data/urtox_clean.jsonl"
python "$HERE/prepare_data.py" --csv "$HERE/../URTOX_v2.csv" --out "$OUT/data" || exit 1

SEEDS=${SEEDS:-"42 123 456 789 1011"}
SEEDS3=${SEEDS3:-"42 123 456"}
TX=${TRAIN_EXTRA:-}   # smoke tests only
EX=${EXPLAIN_EXTRA:-}
DL=${DEADLINE_HOURS:-10.8}
T="python $HERE/train.py --data $DATA --out $OUT $TX --deadline_hours $DL"

lane0() {  # MUTEX, explanations, preprocessing ablation, larger encoder
  CUDA_VISIBLE_DEVICES=0 $T --models xlmr_crf --seeds $SEEDS --save_model
  CUDA_VISIBLE_DEVICES=0 python "$HERE/explain.py" --data "$DATA" --out "$OUT" $EX --seed 42 || echo "explain failed"
  CUDA_VISIBLE_DEVICES=0 $T --models xlmr_crf --variant no_preprocessing --seeds $SEEDS
  CUDA_VISIBLE_DEVICES=0 $T --models xlmr_large_crf --seeds $SEEDS3 --deadline_hours ${DEADLINE_LARGE:-9.9}
}
lane1() {  # baselines, learning curve
  CUDA_VISIBLE_DEVICES=$G1 $T --models xlmr mbert bilstm_crf --seeds $SEEDS
  for F in 0.25 0.5 0.75; do
    CUDA_VISIBLE_DEVICES=$G1 $T --models xlmr_crf --variant frac$F --train_fraction $F --seeds $SEEDS3
  done
}
lane0 > >(sed 's/^/[gpu0] /') 2>&1 &
P0=$!
lane1 > >(sed "s/^/[gpu$G1] /") 2>&1 &
P1=$!
wait $P0 $P1
python "$HERE/analyze.py" --data "$DATA" --out "$OUT"
cd "$OUT" && tar --exclude='*.pt' -czf ../urtox_revision_outputs.tar.gz . && cd - >/dev/null
echo "### done $(date -u)"
