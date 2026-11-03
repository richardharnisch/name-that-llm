"""Train and evaluate logistic regression on cached BPE document features."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)

from name_that_llm.config import (
    DATA_DIR,
    DEFAULT_TOKENIZER,
    DEFAULT_TRAINING,
    TOKENIZER_ROOT,
    TRAIN_OUTPUT_DIR,
    TokenizerConfig,
    TrainingConfig,
    save_config,
)
from name_that_llm.encodings import BagOfWords, TokenIndicator
from name_that_llm.preprocess import preprocess_data
from name_that_llm.tokenizer import Tokenizer


def encode_samples(
    input_ids: pd.Series, encoder: BagOfWords | TokenIndicator
) -> csr_matrix:
    """Build sparse document vectors from saved IDs without tokenizing again."""
    return vstack(
        [csr_matrix(encoder.from_ids(ids)) for ids in input_ids], format="csr"
    )


def save_plots(labels, predictions, classes, report: dict, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=classes,
        normalize="true",
        xticks_rotation=90,
        values_format=".2f",
        cmap="Blues",
        ax=axis,
    )
    axis.set_title("Validation confusion matrix (fraction of each true class)")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, max(4, len(classes) * 0.4)))
    axis.barh(classes, [report[label]["f1-score"] for label in classes])
    axis.invert_yaxis()
    axis.set(xlim=(0, 1), xlabel="F1 score", title="Validation F1 by model")
    figure.tight_layout()
    figure.savefig(output / "per_class_f1.png", dpi=150)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    start = perf_counter()
    config = TrainingConfig(
        encoder=args.encoder,
        max_iter=args.max_iter,
        seed=args.seed,
        limit_per_class=args.limit_per_class,
    )
    tokenizer_config = TokenizerConfig(
        vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    print("Preparing BPE tokenizer and tokenized samples...", flush=True)
    tokenizer_path, sample_directory, reused = preprocess_data(
        directory=args.data_dir,
        tokenizer_root=args.tokenizer_root,
        output=args.tokenized_root,
        vocab_size=tokenizer_config.vocab_size,
        min_frequency=tokenizer_config.min_frequency,
    )
    print(
        f"{'Reused' if reused else 'Created'} BPE tokenizer and samples: {tokenizer_path}",
        flush=True,
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    train = pd.read_parquet(sample_directory / "train.parquet")
    val = pd.read_parquet(sample_directory / "val.parquet")
    if config.limit_per_class is not None:
        train, val = (
            frame.groupby("model", group_keys=False).sample(
                n=min(config.limit_per_class, frame.groupby("model").size().min()),
                random_state=config.seed,
            )
            for frame in (train, val)
        )

    print(f"Train: {len(train):,} rows; validation: {len(val):,} rows", flush=True)
    encoder = (
        BagOfWords(tokenizer)
        if config.encoder == "bag-of-words"
        else TokenIndicator(tokenizer)
    )
    print(f"Encoder: {config.encoder}; vocabulary: {len(encoder):,} tokens", flush=True)
    x_train = encode_samples(train["input_ids"], encoder)
    x_val = encode_samples(val["input_ids"], encoder)

    print("Fitting logistic regression...", flush=True)
    classifier = LogisticRegression(max_iter=config.max_iter, random_state=config.seed)
    classifier.fit(x_train, train["model"])
    predictions = classifier.predict(x_val)
    report = classification_report(
        val["model"],
        predictions,
        labels=classifier.classes_,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        **asdict(config),
        "tokenizer": "bpe",
        "tokenizer_path": str(tokenizer_path),
        "sample_directory": str(sample_directory),
        "requested_vocab_size": tokenizer_config.vocab_size,
        "min_frequency": tokenizer_config.min_frequency,
        "train_rows": len(train),
        "validation_rows": len(val),
        "vocab_size": len(encoder),
        "train_accuracy": classifier.score(x_train, train["model"]),
        "validation_accuracy": accuracy_score(val["model"], predictions),
        "validation_macro_f1": f1_score(val["model"], predictions, average="macro"),
        "validation_log_loss": log_loss(
            val["model"], classifier.predict_proba(x_val), labels=classifier.classes_
        ),
        "elapsed_seconds": perf_counter() - start,
        "classification_report": report,
    }
    for name in (
        "train_accuracy",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_log_loss",
    ):
        print(f"{name}: {metrics[name]:.4f}")
    print(f"Training and evaluation: {metrics['elapsed_seconds']:.1f} seconds")
    print("\nValidation classification report:")
    print(classification_report(val["model"], predictions, digits=3, zero_division=0))

    args.output.mkdir(parents=True, exist_ok=True)
    save_config(
        args.output / "config.json", tokenizer=tokenizer_config, training=config
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    save_plots(val["model"], predictions, classifier.classes_, report, args.output)
    print(
        "Saved config.json, metrics.json, confusion_matrix.png, "
        f"and per_class_f1.png to {args.output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing train.csv and val.csv",
    )
    parser.add_argument("--tokenizer-root", type=Path, default=TOKENIZER_ROOT)
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        help="Tokenized data root (default: <data-dir>/tokenized)",
    )
    parser.add_argument("--output", type=Path, default=TRAIN_OUTPUT_DIR)
    parser.add_argument(
        "--encoder",
        choices=("bag-of-words", "token-indicator"),
        default=DEFAULT_TRAINING.encoder,
    )
    parser.add_argument(
        "--vocab-size",
        "--max-vocab",
        dest="vocab_size",
        type=int,
        default=DEFAULT_TOKENIZER.vocab_size,
        help="Requested BPE vocabulary size",
    )
    parser.add_argument(
        "--min-frequency", type=int, default=DEFAULT_TOKENIZER.min_frequency
    )
    parser.add_argument("--max-iter", type=int, default=DEFAULT_TRAINING.max_iter)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING.seed)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=DEFAULT_TRAINING.limit_per_class,
        help="Cap rows per class in each set for a quick run",
    )
    args = parser.parse_args()
    if args.vocab_size < 2 or args.min_frequency < 1 or args.max_iter < 1:
        parser.error(
            "--vocab-size must be at least 2; --min-frequency and --max-iter must be positive."
        )
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be positive.")
    try:
        run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
