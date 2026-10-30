"""Dataset checks shared by training and prediction."""

from pathlib import Path

import pandas as pd

LABELS = (
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
)
# Do not mix reconstructed competition prompts into validation. The pilot uses
# a different generation protocol; the Kimi-only rerun is not a balanced block.
DEFAULT_SPLITS = ("new", "new2", "proxy_v1")


def validate_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The input has no rows.")
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Column {column!r} contains missing or empty values.")
    if "id" in columns and frame["id"].duplicated().any():
        raise ValueError("The input contains duplicate IDs.")


def load_training_data(directory: Path, splits: tuple[str, ...] = DEFAULT_SPLITS) -> pd.DataFrame:
    path = directory / "passages_clean.csv"
    if not path.is_file():
        raise ValueError(f"Missing {path}. Run `uv run python scripts/download_data.py` first.")
    if not splits or set(splits) - set(DEFAULT_SPLITS):
        raise ValueError(f"Use one or more of these training splits: {DEFAULT_SPLITS}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = ("id", "prompt_id", "text", "model", "split")
    validate_columns(frame, columns)
    frame = frame.loc[frame["split"].isin(splits), list(columns)].reset_index(drop=True)
    validate_columns(frame, columns)
    labels = set(frame["model"])
    if labels != set(LABELS):
        raise ValueError(
            f"Expected all 18 competition labels. Missing: {sorted(set(LABELS) - labels)}; "
            f"unknown: {sorted(labels - set(LABELS))}"
        )
    normalised = frame["text"].str.split().str.join(" ")
    if normalised.duplicated().any():
        raise ValueError("Duplicate passage text could leak across folds. Check the dataset.")
    return frame


def load_test_data(path: Path) -> pd.DataFrame:
    # String dtype preserves IDs such as 00123. Do not treat 'NA' as a missing ID.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_columns(frame, ("id", "text"))
    return frame.loc[:, ["id", "text"]]
