"""Fit a BPE tokenizer once and save tokenized training and validation samples."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from name_that_llm.data import LABELS, validate_columns
from name_that_llm.tokenizer import (
    DEFAULT_MIN_FREQUENCY,
    DEFAULT_VOCAB_SIZE,
    UNK_TOKEN,
    Tokenizer,
    train_tokenizer,
)

COLUMNS = ("id", "prompt_id", "text", "model", "split")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_splits(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames = {}
    for name, path in paths.items():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        validate_columns(frame, COLUMNS)
        if set(frame["model"]) != set(LABELS):
            raise ValueError(f"{path} must contain all 18 model labels and no unknown labels.")
        frames[name] = frame.loc[:, list(COLUMNS)].sort_values("id").reset_index(drop=True)
    for column in ("id", "prompt_id"):
        if set(frames["train"][column]) & set(frames["val"][column]):
            raise ValueError(f"Training and validation share values in {column!r}.")
    texts = pd.concat([frame["text"] for frame in frames.values()], ignore_index=True)
    if texts.str.split().str.join(" ").duplicated().any():
        raise ValueError("Duplicate passage text could leak across splits. Check the dataset.")
    return frames


def _write_samples(frame: pd.DataFrame, tokenizer: Tokenizer, path: Path) -> None:
    encodings = tokenizer.encode_batch(frame["text"].tolist())
    label_ids = {model: index for index, model in enumerate(LABELS)}
    table = pa.table(
        {
            **{column: frame[column].tolist() for column in COLUMNS},
            "input_ids": pa.array(
                [encoding.ids for encoding in encodings], type=pa.list_(pa.int32())
            ),
            "label": pa.array(frame["model"].map(label_ids).tolist(), type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def preprocess_data(
    directory: Path = Path("data"),
    tokenizer_root: Path = Path("artifacts/tokenizers"),
    output: Path | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
) -> tuple[Path, Path, bool]:
    """Return the tokenizer path, sample directory, and whether saved files were reused."""
    if vocab_size < 2 or min_frequency < 1:
        raise ValueError(
            "Vocabulary size must be at least 2 and minimum frequency must be positive."
        )
    sources = {name: directory / f"{name}.csv" for name in ("train", "val")}
    for path in sources.values():
        if not path.is_file():
            raise ValueError(f"Missing {path}. Run `uv run python scripts/download_data.py` first.")
    name = f"bpe-vs-{vocab_size}-mf-{min_frequency}"
    tokenizer_path = tokenizer_root / name / "tokenizer.json"
    sample_directory = (output if output is not None else directory / "tokenized") / name
    outputs = {
        "tokenizer.json": tokenizer_path,
        **{f"{split}.parquet": sample_directory / f"{split}.parquet" for split in sources},
    }
    spec = {
        "format_version": 1,
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "pre_tokenizer": "Metaspace",
        "unk_token": UNK_TOKEN,
        "labels": list(LABELS),
        "tokenizers_version": version("tokenizers"),
        "pyarrow_version": version("pyarrow"),
        "sources_sha256": {path.name: _sha256(path) for path in sources.values()},
    }
    manifest_path = sample_directory / "preprocessing_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = None
        if isinstance(manifest, dict) and manifest.get("spec") == spec:
            if all(path.is_file() for path in outputs.values()):
                hashes = {name: _sha256(path) for name, path in outputs.items()}
                if manifest.get("files_sha256") == hashes:
                    return tokenizer_path, sample_directory, True

    frames = _load_splits(sources)
    tokenizer = train_tokenizer(frames["train"]["text"], vocab_size, min_frequency)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    sample_directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    for split, frame in frames.items():
        _write_samples(frame, tokenizer, outputs[f"{split}.parquet"])
    manifest = {
        "spec": spec,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "rows": {split: len(frame) for split, frame in frames.items()},
        "files_sha256": {name: _sha256(path) for name, path in outputs.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return tokenizer_path, sample_directory, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tokenizer-root", type=Path, default=Path("artifacts/tokenizers"))
    parser.add_argument(
        "--output", type=Path, help="Tokenized data root (default: <data-dir>/tokenized)"
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    args = parser.parse_args()
    try:
        tokenizer_path, sample_directory, reused = preprocess_data(
            args.data_dir, args.tokenizer_root, args.output, args.vocab_size, args.min_frequency
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n")
    print(f"{'Reused' if reused else 'Created'} tokenizer: {tokenizer_path}")
    for split in ("train", "val"):
        print(f"{split}: {sample_directory / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
