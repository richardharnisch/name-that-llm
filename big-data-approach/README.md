# Big data approach

Data: crtproduction's [Name That LLM Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool)
(CC BY-SA 4.0): 212,623 passages from 18 models, median 102 words, 35,573 distinct prompt groups.

The main experiment repeats the small-data test with the same models, the same hyperparameters and the
same pipeline: log-linear, MLP and SVM on the token-indicator encoder (the best encoder in the
small-data runs), 5-fold cross-validation with whole prompts in one fold, and a BPE tokenizer
(vocabulary 8000) fitted per fold on its training part.

## Files

- `data/raw/` holds the Kaggle download (`open-pool.zip`, extracted to `open-pool/`, about 220 MB). Both `big-data-approach/data/` and `big-data-approach/artifacts/` are in the repo `.gitignore`.
- `make_folds.py` writes `data/folds.csv` in the format the small-data pipeline reads (5 folds, grouped by prompt, balanced by label, seed 42). 1,925 passages have no canonical prompt id (`prompt_canon` is empty); they are grouped by the dataset's `prompt_group` instead.
- `run_all.sh` preprocesses nothing; it runs `scripts/train.py` for the three models. Run from the `name-that-llm` folder.
- Results: `artifacts/runs/<model>-token-indicator/cv_metrics.json`, with logs in `artifacts/runs/<model>-token-indicator.log`.

### Order of commands (from the `name-that-llm` folder)

```sh
uv run python big-data-approach/make_folds.py
uv run python scripts/preprocess_data.py --data-dir big-data-approach/data --tokenizer-root big-data-approach/artifacts/tokenizers
big-data-approach/run_all.sh
```

### Training on a lot of data

`train.py` no longer builds a dense documents x vocabulary matrix. Each document is encoded once and
stored sparsely, and each batch is turned dense just before it goes into the model. On the small-data
folds this gives results identical to the old code (checked on fold 0 of log-linear + token-indicator:
all five compared metrics matched exactly).

## Extra: fine-tuned transformer (outside the assignment's models)

`prepare.py` makes a single 80/10/10 prompt-hash split (`data/pool.parquet`) and `finetune_transformer.py`
fine-tunes `distilroberta-base` on it. This is not one of the assignment's models, so it is only an extra
reference point and it uses a different split from the cross-validation above.
Results are in `results_transformer_full.json` (3 epochs, 169,184 training passages, 3.6 hours on an M5):
validation accuracy 0.519, test accuracy 0.534, test macro F1 0.488. Validation accuracy by epoch was
0.466, 0.504, 0.519. A 20k-passage, 1-epoch test is in `results_transformer_20k.json` (test accuracy 0.301).
In `prepare.py` the 1,925 passages without a prompt id all hash to one group and so all fall into the test split.
