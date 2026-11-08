"""Split the Open Pool into train/val/test by prompt, so no prompt appears in two splits."""

import hashlib
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/open-pool/ntl_open_pool.parquet")
OUT = Path("data/pool.parquet")
COLUMNS = ["id", "text", "model", "lab", "prompt_canon", "source_dataset", "n_words"]


def split_of(prompt: str) -> str:
    """Deterministic 80/10/10 split from a hash of the prompt id."""
    bucket = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def main() -> None:
    frame = pd.read_parquet(RAW, columns=COLUMNS)
    frame["split"] = frame["prompt_canon"].map(split_of)
    frame.to_parquet(OUT, index=False)
    print(frame.groupby("split").size().to_string())
    print(frame.groupby(["split", "model"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main()
