"""Cross-validate our own PyTorch classifiers on cached BPE document features."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)

from name_that_llm.classifiers import MLP, LogLinear
from name_that_llm.config import (
    DATA_DIR,
    DEFAULT_MLP,
    DEFAULT_TOKENIZER,
    DEFAULT_TRAINING,
    TOKENIZER_ROOT,
    TRAIN_OUTPUT_DIR,
    MLPConfig,
    TokenizerConfig,
    TrainingConfig,
    save_config,
)
from name_that_llm.data import LABELS, load_folds
from name_that_llm.encodings import TFIDF, BagOfWords, Encoder, TokenIndicator
from name_that_llm.preprocess import preprocess_data
from name_that_llm.tokenizer import Tokenizer

MODELS = ("log-linear", "mlp", "svm")
ENCODERS = ("bag-of-words", "token-indicator", "tf-idf")


def build_encoder(name: str, tokenizer: Tokenizer, train_texts: pd.Series) -> Encoder:
    """Create an encoder. TF-IDF weights come from training texts only."""
    if name == "bag-of-words":
        return BagOfWords(tokenizer)
    if name == "token-indicator":
        return TokenIndicator(tokenizer)
    return TFIDF(tokenizer, train_texts.to_numpy())


def encoder_from_checkpoint(checkpoint: dict, tokenizer: Tokenizer) -> Encoder:
    """Rebuild the encoder a saved model was trained with."""
    name = checkpoint["encoder"]
    if name == "tf-idf":
        return TFIDF.from_idf(tokenizer, checkpoint["idf"].numpy())
    return build_encoder(name, tokenizer, pd.Series(dtype=str))


def encode_samples(input_ids: pd.Series, encoder: Encoder) -> torch.Tensor:
    """Build dense document vectors from saved IDs without tokenizing again."""
    features = np.zeros((len(input_ids), len(encoder)), dtype=np.float32)
    for row, ids in enumerate(input_ids):
        features[row] = encoder.from_ids(ids)
    return torch.from_numpy(features)


def build_model(
    config: TrainingConfig, mlp_config: MLPConfig, d_in: int, n_classes: int
) -> nn.Module:
    if config.model in ("log-linear", "svm"):
        # A linear SVM is the same linear layer as log-linear, trained with a hinge loss.
        return LogLinear(d_in, n_classes)
    return MLP(d_in, n_classes, hidden=mlp_config.hidden, act=getattr(nn, mlp_config.activation))


def build_loss(config: TrainingConfig) -> nn.Module:
    """Multi-class hinge loss for the SVM; cross-entropy for the probabilistic models."""
    return nn.MultiMarginLoss() if config.model == "svm" else nn.CrossEntropyLoss()


@torch.no_grad()
def predict_logits(model: nn.Module, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    return torch.cat([model(batch) for batch in features.split(batch_size)])


def fit(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: TrainingConfig,
) -> list[dict]:
    """Train for a fixed number of epochs. The last epoch is kept, never the best on validation."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = build_loss(config)
    generator = torch.Generator().manual_seed(config.seed)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        total_loss = 0.0
        for batch in order.split(config.batch_size):
            optimizer.zero_grad()
            loss = loss_function(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        val_logits = predict_logits(model, x_val, config.batch_size)
        val_predictions = val_logits.argmax(dim=1).numpy()
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / len(x_train),
            "validation_loss": loss_function(val_logits, y_val).item(),
            "validation_accuracy": accuracy_score(y_val.numpy(), val_predictions),
            "validation_macro_f1": f1_score(y_val.numpy(), val_predictions, average="macro"),
        }
        history.append(entry)
        print(
            f"epoch {epoch:3d}/{config.epochs}  train loss {entry['train_loss']:.4f}  "
            f"val loss {entry['validation_loss']:.4f}  val acc {entry['validation_accuracy']:.4f}  "
            f"val macro F1 {entry['validation_macro_f1']:.4f}",
            flush=True,
        )
    return history


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainingConfig,
    mlp_config: MLPConfig,
    tokenizer_config: TokenizerConfig,
    tokenizer_path: Path,
    encoder: Encoder,
) -> None:
    """Save everything needed to rebuild the model and its input features."""
    extra = {"idf": torch.from_numpy(encoder.idf)} if isinstance(encoder, TFIDF) else {}
    torch.save(
        {
            **extra,
            "state_dict": model.state_dict(),
            "model": config.model,
            "encoder": config.encoder,
            "d_in": len(encoder),
            "labels": list(LABELS),
            "hidden": list(mlp_config.hidden),
            "activation": mlp_config.activation,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer": asdict(tokenizer_config),
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    """Load a saved model in evaluation mode, with its metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = TrainingConfig(model=checkpoint["model"])
    mlp_config = MLPConfig(hidden=tuple(checkpoint["hidden"]), activation=checkpoint["activation"])
    model = build_model(config, mlp_config, checkpoint["d_in"], len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def save_plots(labels, predictions, classes, report: dict, output: Path, n_folds: int) -> None:
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
    axis.set_title(f"Out-of-fold confusion matrix ({n_folds} fold(s); fraction of each true class)")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, max(4, len(classes) * 0.4)))
    axis.barh(classes, [report[label]["f1-score"] for label in classes])
    axis.invert_yaxis()
    axis.set(xlim=(0, 1), xlabel="F1 score", title="Out-of-fold F1 by model")
    figure.tight_layout()
    figure.savefig(output / "per_class_f1.png", dpi=150)
    plt.close(figure)


def save_learning_curves(history: list[dict], output: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    figure, (loss_axis, score_axis) = plt.subplots(1, 2, figsize=(12, 4))
    loss_axis.plot(epochs, [entry["train_loss"] for entry in history], label="train")
    loss_axis.plot(epochs, [entry["validation_loss"] for entry in history], label="validation")
    loss_axis.set(
        xlabel="Epoch", ylabel="Training loss (hinge for SVM, else cross-entropy)", title="Loss"
    )
    loss_axis.legend()
    score_axis.plot(epochs, [entry["validation_accuracy"] for entry in history], label="accuracy")
    score_axis.plot(epochs, [entry["validation_macro_f1"] for entry in history], label="macro F1")
    score_axis.set(xlabel="Epoch", ylabel="Score", title="Validation scores")
    score_axis.legend()
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=150)
    plt.close(figure)


SCORES = ("train_accuracy", "validation_accuracy", "validation_macro_f1", "validation_log_loss")


def run_fold(
    args: argparse.Namespace,
    config: TrainingConfig,
    mlp_config: MLPConfig,
    tokenizer_config: TokenizerConfig,
    fold: int,
    output: Path,
) -> tuple[dict, list[str], list[str]]:
    """Train on every fold except `fold`, validate on `fold`, and save the model."""
    start = perf_counter()
    tokenizer_path, sample_directory, reused = preprocess_data(
        directory=args.data_dir,
        tokenizer_root=args.tokenizer_root,
        output=args.tokenized_root,
        vocab_size=tokenizer_config.vocab_size,
        min_frequency=tokenizer_config.min_frequency,
        fold=fold,
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
    encoder = build_encoder(config.encoder, tokenizer, train["text"])
    print(f"Encoder: {config.encoder}; vocabulary: {len(encoder):,} tokens", flush=True)
    x_train = encode_samples(train["input_ids"], encoder)
    x_val = encode_samples(val["input_ids"], encoder)
    y_train = torch.tensor(train["label"].to_numpy(), dtype=torch.long)
    y_val = torch.tensor(val["label"].to_numpy(), dtype=torch.long)

    torch.manual_seed(config.seed)
    model = build_model(config, mlp_config, len(encoder), len(LABELS))
    print(f"Training {config.model} ({sum(p.numel() for p in model.parameters()):,} parameters)")
    history = fit(model, x_train, y_train, x_val, y_val, config)

    train_logits = predict_logits(model, x_train, config.batch_size)
    val_logits = predict_logits(model, x_val, config.batch_size)
    val_predictions = val_logits.argmax(dim=1).numpy()
    true_names = [LABELS[i] for i in y_val.tolist()]
    predicted_names = [LABELS[i] for i in val_predictions]
    metrics = {
        "fold": fold,
        **asdict(config),
        "mlp": asdict(mlp_config) if config.model == "mlp" else None,
        "tokenizer": "bpe",
        "tokenizer_path": str(tokenizer_path),
        "sample_directory": str(sample_directory),
        "requested_vocab_size": tokenizer_config.vocab_size,
        "min_frequency": tokenizer_config.min_frequency,
        "train_rows": len(train),
        "validation_rows": len(val),
        "vocab_size": len(encoder),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_accuracy": accuracy_score(y_train.numpy(), train_logits.argmax(dim=1).numpy()),
        "validation_accuracy": accuracy_score(y_val.numpy(), val_predictions),
        "validation_macro_f1": f1_score(y_val.numpy(), val_predictions, average="macro"),
        # SVM scores are not probabilities, so log loss is undefined for it.
        "validation_log_loss": None
        if config.model == "svm"
        else log_loss(
            y_val.numpy(),
            torch.softmax(val_logits, dim=1).numpy(),
            labels=list(range(len(LABELS))),
        ),
        "elapsed_seconds": perf_counter() - start,
        "history": history,
    }
    for name in SCORES:
        if metrics[name] is not None:
            print(f"fold {fold} {name}: {metrics[name]:.4f}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_checkpoint(
        output / "model.pt", model, config, mlp_config, tokenizer_config, tokenizer_path, encoder
    )
    save_learning_curves(history, output)
    return metrics, true_names, predicted_names


def summarise(results: list[dict]) -> dict:
    """Mean and sample standard deviation of each score across folds."""
    summary = {}
    for name in SCORES:
        values = [result[name] for result in results]
        if any(value is None for value in values):
            summary[name] = None
            continue
        summary[name] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def run(args: argparse.Namespace) -> None:
    start = perf_counter()
    config = TrainingConfig(
        model=args.model,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        limit_per_class=args.limit_per_class,
    )
    mlp_config = MLPConfig(hidden=tuple(args.hidden), activation=DEFAULT_MLP.activation)
    tokenizer_config = TokenizerConfig(vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    available = sorted(set(load_folds(args.data_dir)["fold"]))
    folds = available if args.fold is None else [args.fold]
    if not set(folds) <= set(available):
        raise ValueError(f"Fold {args.fold} does not exist. Folds: {available}.")

    args.output.mkdir(parents=True, exist_ok=True)
    save_config(
        args.output / "config.json",
        tokenizer=tokenizer_config,
        training=config,
        **({"mlp": mlp_config} if config.model == "mlp" else {}),
    )
    results, all_true, all_predicted = [], [], []
    for position, fold in enumerate(folds, start=1):
        print(f"\n===== Fold {fold} ({position}/{len(folds)}) =====", flush=True)
        metrics, true_names, predicted_names = run_fold(
            args, config, mlp_config, tokenizer_config, fold, args.output / f"fold-{fold}"
        )
        results.append(metrics)
        all_true += true_names
        all_predicted += predicted_names

    # Every sample is validated once, so pooling the folds gives one out-of-fold prediction each.
    report = classification_report(
        all_true, all_predicted, labels=list(LABELS), output_dict=True, zero_division=0
    )
    summary = {
        "folds": folds,
        "model": config.model,
        "encoder": config.encoder,
        "scores": summarise(results),
        "elapsed_seconds": perf_counter() - start,
        "pooled_out_of_fold_classification_report": report,
    }
    (args.output / "cv_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_plots(all_true, all_predicted, list(LABELS), report, args.output, len(folds))

    print(f"\n===== {config.model} + {config.encoder}: {len(folds)}-fold summary =====")
    for name in SCORES:
        score = summary["scores"][name]
        if score is not None:
            print(f"{name:22s} mean {score['mean']:.4f}  std {score['std']:.4f}")
    print(f"Total time: {summary['elapsed_seconds']:.1f} seconds")
    print("\nPooled out-of-fold classification report:")
    print(
        classification_report(
            all_true, all_predicted, labels=list(LABELS), digits=3, zero_division=0
        )
    )
    print(
        f"Saved cv_metrics.json, config.json, confusion_matrix.png, per_class_f1.png, and "
        f"fold-*/(model.pt, metrics.json, learning_curves.png) to {args.output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing folds.csv",
    )
    parser.add_argument("--fold", type=int, help="Run one fold only (default: every fold)")
    parser.add_argument("--tokenizer-root", type=Path, default=TOKENIZER_ROOT)
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        help="Tokenized data root (default: <data-dir>/tokenized)",
    )
    parser.add_argument("--output", type=Path, default=TRAIN_OUTPUT_DIR)
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_TRAINING.model)
    parser.add_argument("--encoder", choices=ENCODERS, default=DEFAULT_TRAINING.encoder)
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=list(DEFAULT_MLP.hidden),
        help="Hidden layer sizes, used only by --model mlp",
    )
    parser.add_argument(
        "--vocab-size",
        dest="vocab_size",
        type=int,
        default=DEFAULT_TOKENIZER.vocab_size,
        help="Requested BPE vocabulary size",
    )
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_TOKENIZER.min_frequency)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAINING.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_TRAINING.weight_decay)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING.seed)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=DEFAULT_TRAINING.limit_per_class,
        help="Cap rows per class in each set for a quick run",
    )
    args = parser.parse_args()
    if args.vocab_size < 2 or args.min_frequency < 1:
        parser.error("--vocab-size must be at least 2 and --min-frequency must be positive.")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("--learning-rate must be positive and --weight-decay must not be negative.")
    if any(size < 1 for size in args.hidden):
        parser.error("--hidden sizes must be positive.")
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be positive.")
    if args.fold is not None and args.fold < 0:
        parser.error("--fold must not be negative.")
    try:
        run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
