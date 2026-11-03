"""Create deterministic train/validation files from an existing download."""

import argparse
from pathlib import Path

from name_that_llm.config import DATA_DIR, DEFAULT_SPLIT
from name_that_llm.data import write_data_splits
from name_that_llm.download import DEFAULT_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_SPLIT.val_size,
        help="Fraction of prompts for validation",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT.seed)
    args = parser.parse_args()
    try:
        train, val = write_data_splits(
            args.input, args.output, args.val_size, args.seed
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Data split failed: {error}\n")
    print(f"Train: {len(train):,} rows -> {args.output / 'train.csv'}")
    print(f"Validation: {len(val):,} rows -> {args.output / 'val.csv'}")


if __name__ == "__main__":
    main()
