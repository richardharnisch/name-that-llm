"""Save and restore classifier checkpoints for the cross-validation experiment."""

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from name_that_llm.classifiers import build_model
from name_that_llm.config import MLPConfig, TokenizerConfig, TrainingConfig
from name_that_llm.data import LABELS
from name_that_llm.encodings import TFIDF, Encoder, L2Normalized


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
    base = encoder.encoder if isinstance(encoder, L2Normalized) else encoder
    extra = {"idf": torch.from_numpy(base.idf)} if isinstance(base, TFIDF) else {}
    torch.save(
        {
            **extra,
            "state_dict": model.state_dict(),
            "model": config.model,
            "encoder": config.encoder,
            "normalize": isinstance(encoder, L2Normalized),
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
    mlp_config = MLPConfig(
        hidden=tuple(checkpoint["hidden"]), activation=checkpoint["activation"]
    )
    model = build_model(
        config, mlp_config, checkpoint["d_in"], len(checkpoint["labels"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint
