#!/usr/bin/env bash
# Full pipeline, ~5-6 h on a 4-core Intel laptop CPU. Each stage can also be run by hand.
#   PY=/path/to/python ./run_all.sh            # everything
#   SKIP_TRAIN=1 ./run_all.sh                  # reuse existing adapters
#   SKIP_OLLAMA=1 ./run_all.sh                 # no larger-model baselines
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONWARNINGS=ignore

$PY make_dataset.py

if [[ -z "${SKIP_TRAIN:-}" ]]; then
  $PY train_lora.py --run main                                         # 300 ex x 2 epochs
  $PY train_lora.py --run overfit --n_train 40 --epochs 8 --grad_accum 4  # overfitting demo
fi

for split in test_id test_ood; do
  $PY evaluate.py --mode zero_shot --split $split
  $PY evaluate.py --mode few_shot  --split $split
  $PY evaluate.py --mode lora --adapter adapters/main/epoch1 --split $split
  $PY evaluate.py --mode lora --adapter adapters/main/epoch2 --split $split
  $PY evaluate.py --mode lora --adapter adapters/overfit/epoch8 --split $split
  if [[ -z "${SKIP_OLLAMA:-}" ]]; then
    $PY evaluate.py --mode ollama:qwen2.5:1.5b --split $split
    $PY evaluate.py --mode ollama:llama3.2:3b --split $split
  fi
done

$PY make_figs.py
