"""Compute Qwen3 backbone features for the passages an experiment selects.

Writes one Parquet file of numeric features keyed on passage id plus metadata.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from name_that_llm.baseline import DEFAULT_SPLITS
from name_that_llm.data import load_training_data
from name_that_llm.download import DEFAULT_DATA_DIR, sha256
from name_that_llm.lm_features import (
    BACKBONE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LAYERS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_DIR,
    generate_lm_features,
)


def extract_features(
    data_dir: Path,
    output: Path,
    *,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    layers: str = DEFAULT_LAYERS,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    frame = load_training_data(data_dir, splits)
    features = generate_lm_features(
        frame[["text"]], model_dir=DEFAULT_MODEL_DIR, layers_spec=layers, max_length=max_length
    )
    table = features.reset_index(names="row_index")
    table.insert(0, "id", frame["id"].to_numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"{output} already exists. Use --to to choose a new location.")
    table.to_parquet(output, index=False)
    report = {
        "backbone": BACKBONE,
        "layers": layers,
        "max_length": max_length,
        "batch_size": batch_size,
        "rows": len(frame),
        "columns": int(table.shape[1] - 2),
        "data_sha256": sha256(data_dir / "passages_clean.csv"),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "file": str(output),
        "file_sha256": sha256(output),
    }
    sidecar = output.with_suffix(".parquet.meta.json")
    sidecar.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--to", type=Path, default=Path("artifacts/features/lm_v2.parquet"))
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    report = extract_features(
        args.from_dir,
        args.to,
        layers=args.layers,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
