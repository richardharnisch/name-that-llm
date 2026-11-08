#!/bin/sh
# Same test as the small-data grid, on the Open Pool: 3 models x token-indicator, 5-fold CV.
# Run from the name-that-llm folder.
for model in log-linear mlp svm; do
  out=big-data-approach/artifacts/runs/$model-token-indicator
  [ -f $out/cv_metrics.json ] && continue
  uv run python scripts/train.py --model $model --encoder token-indicator \
    --data-dir big-data-approach/data --tokenizer-root big-data-approach/artifacts/tokenizers \
    --output $out > $out.log 2>&1 || echo "FAILED $model"
done
echo done
