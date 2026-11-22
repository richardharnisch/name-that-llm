"""Save and restore classifier checkpoints for the cross-validation experiment."""

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from name_that_llm.classifiers import build_model
from name_that_llm.config import (
    DEFAULT_LSTM_HIDDEN_SIZE,
    MLPConfig,
    TokenizerConfig,
    TrainingConfig,
)
from name_that_llm.embeddings import PRETRAINED_MODELS, WORD_TOKENIZER_NAME
from name_that_llm.encodings import TFIDF, L2Normalized
from name_that_llm.utils.data import LABELS
from name_that_llm.utils.encoder import Encoder


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainingConfig,
    mlp_config: MLPConfig,
    tokenizer_config: TokenizerConfig,
    tokenizer_path: Path | None,
    encoder: Encoder,
) -> None:
    """Save everything needed to rebuild the model and its input features."""
    base = encoder.encoder if isinstance(encoder, L2Normalized) else encoder
    if tokenizer_path is None:
        tokenizer_metadata = {
            "tokenizer": WORD_TOKENIZER_NAME,
            "pretrained_model": PRETRAINED_MODELS[config.encoder],
            "pooling": config.pooling or "mean",
        }
        if config.pooling == "lstm":
            tokenizer_metadata["lstm_hidden_size"] = config.lstm_hidden_size
    else:
        tokenizer_metadata = {
            "tokenizer_path": str(tokenizer_path),
            "tokenizer": asdict(tokenizer_config),
        }
    if isinstance(base, TFIDF):
        tokenizer_metadata["idf"] = torch.from_numpy(base.idf)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": config.model,
            "encoder": config.encoder,
            "normalize": (
                config.normalize if tokenizer_path is None else isinstance(encoder, L2Normalized)
            ),
            "d_in": len(encoder),
            "labels": list(LABELS),
            "hidden": list(mlp_config.hidden),
            "activation": mlp_config.activation,
            **tokenizer_metadata,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    """Load a saved model in evaluation mode, with its metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = TrainingConfig(
        model=checkpoint["model"],
        normalize=checkpoint.get("normalize", False),
        pooling=checkpoint.get("pooling"),
        lstm_hidden_size=checkpoint.get("lstm_hidden_size", DEFAULT_LSTM_HIDDEN_SIZE),
    )
    mlp_config = MLPConfig(
        hidden=tuple(checkpoint["hidden"]), activation=checkpoint["activation"]
    )
    model = build_model(
        config, mlp_config, checkpoint["d_in"], len(checkpoint["labels"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint
