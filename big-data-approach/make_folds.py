"""Write 5 prompt-grouped, label-balanced cross-validation folds for the Open Pool.

Output is data/folds.csv in the same format the small-data pipeline reads, so train.py and
preprocess.py run on it unchanged (pass --data-dir big-data-approach/data).
Run from the name-that-llm folder: uv run python big-data-approach/make_folds.py
"""

from pathlib import Path

import pandas as pd

from name_that_llm.config import DEFAULT_SPLIT, SplitConfig, save_config
from name_that_llm.data import assign_folds, fold_summary

RAW = Path("big-data-approach/data/raw/open-pool/ntl_open_pool.parquet")
OUT = Path("big-data-approach/data")

pool = pd.read_parquet(RAW, columns=["id", "prompt_canon", "prompt_group", "text", "model"])
# 1,925 passages have no canonical prompt id. Group those by the source's own prompt_group
# instead, so that they do not all fall into one group (and one fold).
missing = pool["prompt_canon"].fillna("").str.strip().eq("")
pool["prompt_id"] = pool["prompt_canon"].where(~missing, "group:" + pool["prompt_group"])
frame = pool[["id", "prompt_id", "text", "model"]].assign(split="open_pool")
frame = assign_folds(frame, DEFAULT_SPLIT.n_folds, DEFAULT_SPLIT.seed)
frame.to_csv(OUT / "folds.csv", index=False, encoding="utf-8", lineterminator="\n")
save_config(OUT / "split_config.json", split=SplitConfig(splits=("open_pool",)))
print(len(frame), "rows;", fold_summary(frame))
