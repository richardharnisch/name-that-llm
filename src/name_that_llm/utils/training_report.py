"""Plot learning curves and report cross-validation scores."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)

from name_that_llm.config import MLPConfig, TrainingConfig
from name_that_llm.utils.data import LABELS

SCORES = (
    "train_accuracy",
    "validation_accuracy",
    "validation_macro_f1",
    "validation_log_loss",
)


def save_plots(
    labels, predictions, classes, report: dict, output: Path, n_folds: int
) -> None:
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
    axis.set_title(
        f"Out-of-fold confusion matrix ({n_folds} fold(s); fraction of each true class)"
    )
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
    loss_axis.plot(
        epochs, [entry["validation_loss"] for entry in history], label="validation"
    )
    loss_axis.set(
        xlabel="Epoch",
        ylabel="Training loss (hinge for SVM, else cross-entropy)",
        title="Loss",
    )
    loss_axis.legend()
    score_axis.plot(
        epochs, [entry["validation_accuracy"] for entry in history], label="accuracy"
    )
    score_axis.plot(
        epochs, [entry["validation_macro_f1"] for entry in history], label="macro F1"
    )
    score_axis.set(xlabel="Epoch", ylabel="Score", title="Validation scores")
    score_axis.legend()
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=150)
    plt.close(figure)


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


def score_fold(
    train_logits: torch.Tensor,
    y_train: torch.Tensor,
    val_logits: torch.Tensor,
    y_val: torch.Tensor,
    model_name: str,
) -> tuple[dict[str, float | None], list[str], list[str]]:
    """Return scores and validation labels for one fold."""
    val_predictions = val_logits.argmax(dim=1).numpy()
    true_names = [LABELS[i] for i in y_val.tolist()]
    predicted_names = [LABELS[i] for i in val_predictions]
    scores = {
        "train_accuracy": accuracy_score(y_train.numpy(), train_logits.argmax(dim=1).numpy()),
        "validation_accuracy": accuracy_score(y_val.numpy(), val_predictions),
        "validation_macro_f1": f1_score(y_val.numpy(), val_predictions, average="macro"),
        # SVM scores are not probabilities, so log loss is undefined for it.
        "validation_log_loss": (
            None
            if model_name == "svm"
            else log_loss(
                y_val.numpy(),
                torch.softmax(val_logits, dim=1).numpy(),
                labels=list(range(len(LABELS))),
            )
        ),
    }
    return scores, true_names, predicted_names


def save_fold_metrics(
    output: Path,
    fold: int,
    config: TrainingConfig,
    mlp_config: MLPConfig,
    *,
    metadata: dict,
    scores: dict[str, float | None],
    history: list[dict],
    converged: bool,
    elapsed_seconds: float,
) -> dict:
    """Write the fold metrics and print its scores."""
    metrics = {
        "fold": fold,
        **asdict(config),
        "mlp": asdict(mlp_config) if config.model == "mlp" else None,
        **metadata,
        **scores,
        "epochs_run": len(history),
        "converged": converged,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
    }
    for name in SCORES:
        if metrics[name] is not None:
            print(f"fold {fold} {name}: {metrics[name]:.4f}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def save_cv_report(
    output: Path,
    folds: list[int],
    results: list[dict],
    all_true: list[str],
    all_predicted: list[str],
    config: TrainingConfig,
    start: float,
) -> None:
    """Save and print the aggregate cross-validation results."""
    # Every sample is validated once, so pooling the folds gives one out-of-fold prediction each.
    report = classification_report(
        all_true, all_predicted, labels=list(LABELS), output_dict=True, zero_division=0
    )
    summary = {
        "folds": folds,
        "model": config.model,
        "encoder": config.encoder,
        "scores": summarise(results),
        "epochs_run": [result["epochs_run"] for result in results],
        "converged": [result["converged"] for result in results],
        "elapsed_seconds": perf_counter() - start,
        "pooled_out_of_fold_classification_report": report,
    }
    (output / "cv_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_plots(all_true, all_predicted, list(LABELS), report, output, len(folds))

    print(f"\n===== {config.model} + {config.encoder}: {len(folds)}-fold summary =====")
    for name in SCORES:
        score = summary["scores"][name]
        if score is not None:
            print(f"{name:22s} mean {score['mean']:.4f}  std {score['std']:.4f}")
    print(
        f"Converged in {sum(summary['converged'])}/{len(folds)} folds; "
        f"epochs run: {summary['epochs_run']} (cap {config.max_epochs})"
    )
    print(f"Total time: {summary['elapsed_seconds']:.1f} seconds")
    print("\nPooled out-of-fold classification report:")
    print(
        classification_report(
            all_true, all_predicted, labels=list(LABELS), digits=3, zero_division=0
        )
    )
    print(
        f"Saved cv_metrics.json, config.json, confusion_matrix.png, per_class_f1.png, and "
        f"fold-*/(model.pt, metrics.json, learning_curves.png) to {output}"
    )
