"""Plot learning curves and report cross-validation scores."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

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
