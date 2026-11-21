"""Train and evaluate PyTorch classifiers on prepared features."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from name_that_llm.classifiers import build_loss
from name_that_llm.config import TrainingConfig

FeatureMatrix = np.ndarray | sp.csr_matrix


def dense_batch(matrix: FeatureMatrix, rows: np.ndarray | slice) -> torch.Tensor:
    """Gather the chosen rows, converting sparse features one batch at a time."""
    batch = matrix[rows]
    return torch.from_numpy(batch.toarray() if sp.issparse(batch) else batch)


@torch.no_grad()
def predict_logits(
    model: nn.Module, features: FeatureMatrix, batch_size: int
) -> torch.Tensor:
    model.eval()
    return torch.cat(
        [
            model(dense_batch(features, slice(start, start + batch_size)))
            for start in range(0, features.shape[0], batch_size)
        ]
    )


def fit(
    model: nn.Module,
    x_train: FeatureMatrix,
    y_train: torch.Tensor,
    x_val: FeatureMatrix,
    y_val: torch.Tensor,
    config: TrainingConfig,
) -> tuple[list[dict], bool]:
    """Train until the training loss converges, or for max_epochs.

    Return the history and whether training converged. The last epoch is kept. The
    validation data is evaluated for the learning curves only. It never decides when to stop.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = build_loss(config)
    generator = torch.Generator().manual_seed(config.seed)
    history = []
    best_loss = float("inf")
    stalled = 0
    converged = False
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], generator=generator)
        total_loss = 0.0
        for batch in order.split(config.batch_size):
            optimizer.zero_grad()
            loss = loss_function(
                model(dense_batch(x_train, batch.numpy())), y_train[batch]
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        val_logits = predict_logits(model, x_val, config.batch_size)
        val_predictions = val_logits.argmax(dim=1).numpy()
        entry = {
            "epoch": epoch,
            "train_loss": total_loss / x_train.shape[0],
            "validation_loss": loss_function(val_logits, y_val).item(),
            "validation_accuracy": accuracy_score(y_val.numpy(), val_predictions),
            "validation_macro_f1": f1_score(
                y_val.numpy(), val_predictions, average="macro"
            ),
        }
        history.append(entry)
        if config.tol > 0:
            # Compare with the best loss so far, not with the previous epoch. The loss of
            # each epoch is noisy, so a comparison of consecutive epochs never settles.
            if entry["train_loss"] < best_loss * (1 - config.tol):
                best_loss, stalled = entry["train_loss"], 0
            else:
                stalled += 1
            converged = stalled >= config.patience
        if epoch == 1 or epoch % 10 == 0 or converged or epoch == config.max_epochs:
            print(
                f"epoch {epoch:3d}/{config.max_epochs}  train loss {entry['train_loss']:.4f}  "
                f"val loss {entry['validation_loss']:.4f}  "
                f"val acc {entry['validation_accuracy']:.4f}  "
                f"val macro F1 {entry['validation_macro_f1']:.4f}",
                flush=True,
            )
        if converged:
            break
    return history, converged


def main() -> None:
    """Run the cross-validation command."""
    from name_that_llm.utils.training_command import main as command_main

    command_main()


if __name__ == "__main__":
    main()
