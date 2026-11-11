#!/usr/bin/env bash
# One-command reproduction of every result in the revised manuscript.
# Usage:  bash revision/run_all.sh [OUT_DIR]      (default: outputs)
# Safe to re-run: finished runs are skipped. Runs are only STARTED while the
# elapsed time is below the deadline, so a 12-hour Kaggle session is never exceeded.
set -euo pipefail
OUT=${1:-outputs}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export PIPELINE_T0=${PIPELINE_T0:-$(date +%s)}
export PYTHONUNBUFFERED=1
LOG="$OUT/pipeline_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo "### start $(date -u) on $(python -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")')"
python -c "import torch, transformers, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__)"

DATA="$OUT/data/urtox_clean.jsonl"
python "$HERE/prepare_data.py" --csv "$HERE/../URTOX_v2.csv" --out "$OUT/data"

SEEDS=${SEEDS:-"42 123 456 789 1011"}
TX=${TRAIN_EXTRA:-}   # smoke tests only
EX=${EXPLAIN_EXTRA:-}
# 1. main comparison (MUTEX first; its seed-42 checkpoint is kept for explanations)
python "$HERE/train.py" --data "$DATA" --out "$OUT" $TX --models xlmr_crf --seeds $SEEDS --save_model --deadline_hours 10.8
python "$HERE/explain.py" --data "$DATA" --out "$OUT" $EX --seed 42 || echo "explain failed"
python "$HERE/train.py" --data "$DATA" --out "$OUT" $TX --models xlmr mbert bilstm_crf --seeds $SEEDS --deadline_hours 10.8
python "$HERE/analyze.py" --data "$DATA" --out "$OUT" || echo "analyze failed"
# 2. preprocessing ablation (raw tokens, no normalisation)
python "$HERE/train.py" --data "$DATA" --out "$OUT" $TX --models xlmr_crf --variant no_preprocessing --seeds $SEEDS --deadline_hours 10.8
# 3. learning curve
for F in 0.25 0.5 0.75; do
  python "$HERE/train.py" --data "$DATA" --out "$OUT" $TX --models xlmr_crf --variant frac$F --train_fraction $F --seeds ${SEEDS3:-42 123 456} --deadline_hours 10.8
done
python "$HERE/analyze.py" --data "$DATA" --out "$OUT" || echo "analyze failed"
# 4. larger encoder (optional; only started if time remains)
python "$HERE/train.py" --data "$DATA" --out "$OUT" $TX --models xlmr_large_crf --seeds ${SEEDS3:-42 123 456} --batch_size 16 --deadline_hours 10.2
python "$HERE/analyze.py" --data "$DATA" --out "$OUT"
# drop the 1 GB checkpoint copy from the archive, keep everything else
cd "$OUT" && tar --exclude='*.pt' -czf ../urtox_revision_outputs.tar.gz . && cd - >/dev/null
echo "### done $(date -u)"
