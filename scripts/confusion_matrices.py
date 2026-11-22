"""Draw confusion matrices from saved out-of-fold predictions, without retraining.

Each system is NAME=PATH, where PATH is a predictions .parquet from
`scripts/embedding_sweep.py --save-predictions`, or a Part 1 run directory with
fold-K/model.pt. Rows are normalized: each row shows how the passages of one true model
were classified. A second plot groups the 18 models into their families.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score  # noqa: E402

from name_that_llm.error_analysis import load_predictions, parse_system  # noqa: E402
from name_that_llm.utils.data import LABELS  # noqa: E402

FAMILIES = ("claude", "gemini", "gpt", "glm", "kimi", "qwen")


def family(label: str) -> str:
    return next(name for name in FAMILIES if label.startswith(name))


def plot(labels, predictions, classes, title: str, path: Path, size: float) -> None:
    figure, axis = plt.subplots(figsize=(size, size * 0.86))
    ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        labels=classes,
        normalize="true",
        xticks_rotation=90,
        values_format=".2f",
        cmap="Blues",
        ax=axis,
        colorbar=False,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", type=parse_system, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/confusion"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, source in args.system:
        frame = load_predictions(source)
        accuracy = accuracy_score(frame["label"], frame["predicted"])
        plot(
            frame["label"],
            frame["predicted"],
            list(LABELS),
            f"{name}: out-of-fold confusion (row-normalized), accuracy {accuracy:.1%}",
            args.output / f"{name}.png",
            14,
        )
        true_family, predicted_family = frame["label"].map(family), frame["predicted"].map(family)
        family_accuracy = accuracy_score(true_family, predicted_family)
        plot(
            true_family,
            predicted_family,
            list(FAMILIES),
            f"{name}: model families, accuracy {family_accuracy:.1%}",
            args.output / f"{name}-families.png",
            6,
        )
        print(f"{name}: model accuracy {accuracy:.4f}, family accuracy {family_accuracy:.4f}")


if __name__ == "__main__":
    main()
