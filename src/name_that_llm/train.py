"""Run a simple tokenizer + encoder + logistic regression experiment."""

import argparse
import json
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

from name_that_llm.data import DEFAULT_SEED, validate_columns
from name_that_llm.encodings import BagOfWords, Encoder, TokenIndicator
from name_that_llm.test_tokenizer import Tokenizer


def encode_texts(texts: pd.Series, encoder: Encoder) -> csr_matrix:
    """Keep document vectors sparse so the full matrix fits in memory."""
    return vstack([csr_matrix(encoder(text)) for text in texts], format="csr")


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
    train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
    val = pd.read_csv(args.val, dtype=str, keep_default_na=False)
    for frame in (train, val):
        validate_columns(frame, ("text", "model"))
    if train["model"].nunique() < 2:
        raise ValueError("Training requires at least two model labels.")
    if set(train["model"]) != set(val["model"]):
        raise ValueError("Train and validation must contain the same model labels.")
    if "prompt_id" in train and "prompt_id" in val:
        if set(train["prompt_id"]) & set(val["prompt_id"]):
            raise ValueError("Train and validation contain overlapping prompt IDs.")
    if args.limit_per_class is not None:
        train, val = (
            frame.groupby("model", group_keys=False).sample(
                n=min(args.limit_per_class, frame.groupby("model").size().min()),
                random_state=args.seed,
            )
            for frame in (train, val)
        )

    print(f"Train: {len(train):,} rows; validation: {len(val):,} rows", flush=True)
    tokenizer = Tokenizer(max_vocab=args.max_vocab).fit(train["text"])
    encoder = BagOfWords(tokenizer) if args.encoder == "bag-of-words" else TokenIndicator(tokenizer)
    print(f"Encoder: {args.encoder}; vocabulary: {len(encoder):,} tokens", flush=True)
    x_train = encode_texts(train["text"], encoder)
    x_val = encode_texts(val["text"], encoder)

    print("Fitting logistic regression...", flush=True)
    classifier = LogisticRegression(max_iter=args.max_iter, random_state=args.seed)
    classifier.fit(x_train, train["model"])
    predictions = classifier.predict(x_val)
    report = classification_report(
        val["model"], predictions, labels=classifier.classes_, output_dict=True, zero_division=0
    )
    metrics = {
        "encoder": args.encoder,
        "seed": args.seed,
        "max_vocab": args.max_vocab,
        "max_iter": args.max_iter,
        "limit_per_class": args.limit_per_class,
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
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    save_plots(val["model"], predictions, classifier.classes_, report, args.output)
    print(f"Saved metrics.json, confusion_matrix.png, and per_class_f1.png to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/simple_train"))
    parser.add_argument(
        "--encoder", choices=("bag-of-words", "token-indicator"), default="bag-of-words"
    )
    parser.add_argument("--max-vocab", type=int, default=5000)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--limit-per-class", type=int, help="Cap rows per class in each set for a quick run"
    )
    args = parser.parse_args()
    if args.max_vocab < 2 or args.max_iter < 1:
        parser.error("--max-vocab must be at least 2 and --max-iter must be positive.")
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be positive.")
    try:
        run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
