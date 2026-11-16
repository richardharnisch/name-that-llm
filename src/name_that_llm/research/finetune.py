"""Train an encoder on one outer fold and export its final embeddings."""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from name_that_llm.data import LABELS, load_folds, split_fold
from name_that_llm.research.groups import validate_design
from name_that_llm.research.run import digest, record

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class EncoderClassifier(nn.Module):
    """Apply masked mean pooling, L2 normalization, and an 18-class linear head."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, len(LABELS))

    def embed(self, batch):
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=1)

    def forward(self, batch):
        return self.head(self.embed(batch))


def tokenize(tokenizer, texts, max_length, device):
    return {
        key: value.to(device)
        for key, value in tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).items()
    }


def train_model(
    model,
    tokenizer,
    train,
    *,
    device,
    epochs=3,
    batch_size=16,
    max_length=256,
    learning_rate=2e-5,
    head_learning_rate=1e-3,
    weight_decay=0.01,
    seed=42,
    max_steps=None,
):
    """Optimize only training rows for a fixed epoch count, without evaluation."""
    y = np.array([LABELS.index(label) for label in train.model])
    counts = np.bincount(y, minlength=len(LABELS))
    if (counts == 0).any():
        raise ValueError("Training requires all author labels.")
    weights = torch.tensor(len(y) / (len(LABELS) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": learning_rate},
            {"params": model.head.parameters(), "lr": head_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(seed)
    steps, completed = 0, 0
    model.train()
    for epoch in range(epochs):
        started = time.perf_counter()
        losses = []
        indices = rng.permutation(len(train))
        for start in range(0, len(indices), batch_size):
            if max_steps is not None and steps >= max_steps:
                return completed, steps
            ix = indices[start : start + batch_size]
            batch = tokenize(tokenizer, train.text.iloc[ix], max_length, device)
            targets = torch.tensor(y[ix], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
        completed = epoch + 1
        print(
            f"Epoch {completed}/{epochs}; steps={steps}; loss={np.mean(losses):.6f}; "
            f"rows/sec={len(train) / (time.perf_counter() - started):.2f}",
            flush=True,
        )
    return completed, steps


def encode(model, tokenizer, frame, *, batch_size, max_length, device):
    model.eval()
    vectors, probabilities = [], []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = tokenize(
                tokenizer, frame.text.iloc[start : start + batch_size], max_length, device
            )
            features = model.embed(batch)
            vectors.append(features.cpu().numpy())
            probabilities.append(model.head(features).softmax(dim=1).cpu().numpy())
    return np.concatenate(vectors), np.concatenate(probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, help="Diagnostic step limit")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new directory.")
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.max_length < 2:
        parser.error("Use positive epochs, batch size, threads and length at least two.")
    if (
        not np.isfinite(args.head_learning_rate)
        or args.head_learning_rate <= 0
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not np.isfinite(args.weight_decay)
        or args.weight_decay < 0
        or (args.max_steps is not None and args.max_steps < 1)
    ):
        parser.error("Invalid learning rate, weight decay, or maximum steps.")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", args.revision):
        parser.error("Use a pinned 40-character Hugging Face commit revision.")
    frame = load_folds(args.data_dir).sort_values("id").reset_index(drop=True)
    validate_design(frame)
    train, validation = split_fold(frame, args.fold)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, cache_dir=args.cache_dir, local_files_only=True
    )
    encoder = AutoModel.from_pretrained(
        args.model, revision=args.revision, cache_dir=args.cache_dir, local_files_only=True
    )
    model = EncoderClassifier(encoder).to(device)
    truncated = 0
    for start in range(0, len(frame), args.batch_size):
        lengths = tokenizer(
            frame.text.iloc[start : start + args.batch_size].tolist(),
            truncation=False,
            return_length=True,
        )["length"]
        truncated += sum(length > args.max_length for length in lengths)
    print(f"Truncated rows: {truncated}/{len(frame)}", flush=True)
    args.output.mkdir(parents=True)
    manifest = dict(
        schema_version=1,
        status="running",
        fold=args.fold,
        folds_sha256=digest(args.data_dir / "folds.csv"),
        train_ids=train.id.tolist(),
        validation_ids=validation.id.tolist(),
        model=args.model,
        revision=args.revision,
        epochs_requested=args.epochs,
        epochs_completed=0,
        completed_training=False,
        seed=args.seed,
        pooling="masked_mean_l2",
        truncated_rows=truncated,
        device=device,
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completed, steps = train_model(
        model,
        tokenizer,
        train,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    embeddings, probabilities = encode(
        model,
        tokenizer,
        frame,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    embeddings_path = args.output / "embeddings.npz"
    np.savez_compressed(embeddings_path, ids=frame.id.to_numpy(dtype=str), embeddings=embeddings)
    predictions_path = args.output / "classifier_predictions.parquet"
    record(
        validation, probabilities[frame.fold.to_numpy() == args.fold], "finetuned_head"
    ).to_parquet(predictions_path, index=False)
    model.cpu()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": args.model,
            "revision": args.revision,
            "labels": LABELS,
            "encoder_config": encoder.config.to_dict(),
        },
        args.output / "checkpoint.pt",
    )
    manifest.update(
        status="complete" if completed == args.epochs else "incomplete",
        epochs_completed=completed,
        completed_training=completed == args.epochs,
        steps_completed=steps,
        embeddings_sha256=digest(embeddings_path),
        classifier_predictions_sha256=digest(predictions_path),
        checkpoint="checkpoint.pt",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {args.output}; completed_training={manifest['completed_training']}", flush=True)


if __name__ == "__main__":
    main()
