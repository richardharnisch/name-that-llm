# Name That LLM

## Prepare the data

```sh
uv run python scripts/download_data.py
```

This downloads and verifies the public dataset, then creates:

- `data/train.csv`: training passages.
- `data/val.csv`: validation passages.
- `data/raw/extended-v7/`: unchanged source files.

For an existing download, create the split without downloading again:

```sh
uv run python scripts/split_data.py
```

The default split uses 80% of prompts for training and 20% for validation, with
seed `42`. All passages with the same `prompt_id` stay in one set. The row ratio
can differ slightly because prompts have different numbers of passages. Both
sets must contain all 18 model labels.

The same dataset version, settings, and locked dependencies produce the same
files on each machine, even if the source CSV row order changes. Output rows are
sorted by ID and use UTF-8 with LF line endings. Repeat runs replace the generated
CSV files. Source files do not change.

Only source groups `new`, `new2`, and `proxy_v1` are included. The CSV `split`
column keeps these source group names; the file name identifies train or validation.
Use only `train.csv` to fit models and preprocessing; use `val.csv` for evaluation.

To change the validation fraction or seed:

```sh
uv run ntl-split --val-size 0.2 --seed 42
```

## Run a simple training experiment

From the project root, after preparing the data:

```sh
uv run python scripts/train.py
```

You can also use `uv run ntl-train` or `uv run python -m name_that_llm.train`.

The pipeline uses the temporary tokenizer in `src/name_that_llm/test_tokenizer.py`,
the existing bag-of-words encoder, and a logistic regression classifier. The
tokenizer lowercases text and separates words and punctuation. It learns up to
5,000 vocabulary entries from training text only, including an unknown-token entry.
Validation text uses this fixed vocabulary.

The script prints row counts, vocabulary size, training accuracy, validation
accuracy, macro F1, log loss, elapsed time, and per-class precision, recall, and F1.
It saves these files to `artifacts/simple_train/`:

- `metrics.json`: scores, settings, and the per-class report.
- `confusion_matrix.png`: validation confusion matrix, normalized by true class.
- `per_class_f1.png`: validation F1 for each model.

For a quick check with the token-indicator encoder:

```sh
uv run python scripts/train.py --encoder token-indicator --limit-per-class 20 --output artifacts/quick
```

`--limit-per-class` samples at most that many rows per class from each set with
the selected seed. Use `--train`, `--val`, `--max-vocab`, `--max-iter`, `--seed`,
and `--output` to change the inputs and run settings. Input CSVs need `text` and
`model` columns. If both files have `prompt_id`, overlapping prompts are rejected.
