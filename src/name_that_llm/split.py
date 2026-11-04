"""Create deterministic cross-validation folds from an existing download."""

import argparse
from pathlib import Path

from name_that_llm.config import DATA_DIR, DEFAULT_SPLIT
from name_that_llm.data import fold_summary, write_data_splits
from name_that_llm.download import DEFAULT_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_SPLIT.n_folds,
        help="Number of cross-validation folds",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT.seed)
    args = parser.parse_args()
    try:
        frame = write_data_splits(args.input, args.output, args.folds, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Data split failed: {error}\n")
    print(f"{len(frame):,} rows, {fold_summary(frame)} -> {args.output / 'folds.csv'}")


if __name__ == "__main__":
    main()
