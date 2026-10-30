"""Frozen Qwen3 backbone features for the Name That LLM competition.

Extracts, per passage:
- mean-pooled decoder states from selected layers
- max-abs-pooled decoder states from selected layers
- token count and token length statistics
- last hidden state statistics (obvious early syntax signal)

Inference only: no fine-tuning, no training data, no Kaggle calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

BACKBONE = "Qwen/Qwen3-1.7B-Base"
DEFAULT_MODEL_DIR = Path("models/qwen3-1.7b-base")
DEFAULT_LAYERS = "1,7,13,19,25,28"
DEFAULT_MAX_LENGTH = 384
DEFAULT_BATCH_SIZE = 32


def parse_layers(spec: str, num_layers: int) -> list[int]:
    """Parse a comma-separated 1-based layer list for hidden_states indexing."""
    layers = [int(part) for part in spec.split(",") if part.strip()]
    if not layers:
        raise ValueError("Layer list is empty.")
    if layers[0] < 1 or layers[-1] > num_layers:
        raise ValueError(f"layers must be in [1..{num_layers}], got {layers}")
    if len(set(layers)) != len(layers):
        raise ValueError(f"duplicate layer indices in {spec}")
    return layers


def _masked_mean(states: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Token-masked mean over the sequence axis: [seq, hidden] -> [hidden]."""
    summed = (states * mask.unsqueeze(-1)).sum(dim=1)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return (summed / count).float().numpy().astype(np.float32)


def _masked_max_abs(states: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Token-masked max absolute activation over the sequence axis."""
    masked = (states.abs() * mask.unsqueeze(-1)).amax(dim=1)
    return masked.float().numpy().astype(np.float32)


@torch.inference_mode()
def generate_lm_features(
    frame: pd.DataFrame,
    model_dir: Path,
    layers_spec: str = DEFAULT_LAYERS,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Run inference over frame[text] and return numeric features indexed frame-style.

    hidden_states has one entry per decoder layer plus the embedding layer:
    hidden_states[0] = embeddings; hidden_states[1..num_hidden_layers] = decoder outputs.
    """
    if frame.empty:
        raise ValueError("No passages to encode.")
    if frame["text"].str.strip().eq("").any():
        raise ValueError("Empty passage text cannot be encoded.")
    torch.set_num_threads(os.cpu_count() or 1)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model = model.to("cpu")
    model.eval()
    num_layers = int(model.config.num_hidden_layers)
    selected = parse_layers(layers_spec, num_layers)

    collected: list[dict[str, np.ndarray]] = []
    for start in range(0, len(frame), batch_size):
        texts = frame.iloc[start : start + batch_size]["text"].tolist()
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        mask = encoded["attention_mask"].to(torch.bfloat16)
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        )
        mean_block: list[np.ndarray] = []
        maxabs_block: list[np.ndarray] = []
        decoder_sum: torch.Tensor | None = None
        for layer_index, states in enumerate(outputs.hidden_states):
            if 1 <= layer_index <= num_layers:
                decoder_sum = states if decoder_sum is None else decoder_sum + states
            if layer_index in selected:
                mean_block.append(_masked_mean(states, mask))
                maxabs_block.append(_masked_max_abs(states, mask))
        mean_states = decoder_sum / float(num_layers)
        if mean_states is None:
            raise ValueError("No decoder states were collected; check the model config.")
        norms = torch.linalg.vector_norm(mean_states.float(), dim=-1)
        row_norm = _masked_mean(norms.unsqueeze(-1), mask).reshape(-1)

        collected.append(
            {
                "mean": np.concatenate(mean_block, axis=1),
                "maxabs": np.concatenate(maxabs_block, axis=1),
                "numeric": np.concatenate(
                    [
                        encoded["attention_mask"].sum(dim=1).float().cpu().numpy()[:, None],
                        row_norm[:, None],
                    ],
                    axis=1,
                ),
            }
        )

    mean = np.concatenate([item["mean"] for item in collected], axis=0).astype(np.float32)
    maxabs = np.concatenate([item["maxabs"] for item in collected], axis=0).astype(np.float32)
    numeric = np.concatenate([item["numeric"] for item in collected], axis=0).astype(np.float32)
    hidden_size = mean.shape[1] // len(selected)
    names = [
        *(f"mean_l{layer}_d{dim}" for layer in selected for dim in range(hidden_size)),
        *(f"maxabs_l{layer}_d{dim}" for layer in selected for dim in range(hidden_size)),
        "token_count",
        "mean_state_norm",
    ]
    merged = np.concatenate([mean, maxabs, numeric], axis=1)
    expected = len(frame)
    if merged.shape != (expected, len(names)):
        raise ValueError(f"Feature shape {merged.shape} does not match {expected} passages.")
    return pd.DataFrame(merged, columns=names, index=frame.index)
