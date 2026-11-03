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
