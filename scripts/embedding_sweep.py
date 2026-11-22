"""Compare word embedding classifiers on the project's prompt-grouped folds."""

import argparse
import gzip
import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from gensim import downloader
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F

from name_that_llm.config import DATA_DIR, DEFAULT_SEED, DEFAULT_TRAINING
from name_that_llm.embeddings import (
    PRETRAINED_MODELS,
    WORD_TOKENIZER_NAME,
    build_embedding_encoder,
    tokenize_embedding_words,
)
from name_that_llm.train import TrainingLossStopper
from name_that_llm.utils.data import LABELS, assign_folds, load_folds

MODELS = {
    "glove300": PRETRAINED_MODELS["glove"],
    "glove50": "glove-wiki-gigaword-50",
    "fasttext300": PRETRAINED_MODELS["fasttext"],
}
INPUT_CACHE = "subword-inputs.npz"


def read_vectors(name: str, word_ids: dict[bytes, int]) -> np.ndarray:
    if name == PRETRAINED_MODELS["fasttext"]:
        encoder = build_embedding_encoder("fasttext")
        table = np.zeros((len(word_ids) + 2, len(encoder)), dtype=np.float32)
        table[list(word_ids.values())] = encoder.from_words([word.decode() for word in word_ids])
        return table
    path = Path(downloader.BASE_DIR) / name / f"{name}.gz"
    if not path.is_file():
        path = Path(downloader.load(name, return_path=True))
    with gzip.open(path, "rb") as source:
        _, width = map(int, next(source).split())
        table = np.zeros((len(word_ids) + 2, width), dtype=np.float32)
        for line in source:
            word, _, values = line.partition(b" ")
            token_id = word_ids.get(word)
            if token_id is not None:
                table[token_id] = np.fromstring(values, dtype=np.float32, sep=" ")
    return table


def prepare(data_dir: Path, output: Path) -> Path:
    passages = load_folds(data_dir).sort_values("id").reset_index(drop=True)
    documents = [tokenize_embedding_words(text) for text in passages["text"]]
    vocabulary = sorted({token for document in documents for token in document})
    indices = {token: i for i, token in enumerate(vocabulary, start=2)}
    word_ids = {word.encode(): index for word, index in indices.items()}
    lengths = np.asarray([max(1, len(document)) for document in documents], dtype=np.int64)
    token_ids = np.zeros((len(documents), max(lengths)), dtype=np.int32)
    for row, document in enumerate(documents):
        token_ids[row, : lengths[row]] = [indices[token] for token in document] or [1]
    tables = {}
    for key, name in MODELS.items():
        print(f"Preparing {name}", flush=True)
        tables[key] = read_vectors(name, word_ids)
    output.mkdir(parents=True, exist_ok=True)
    cache = output / INPUT_CACHE
    with TemporaryDirectory(dir=output) as directory:
        pending = Path(directory) / INPUT_CACHE
        np.savez(
            pending,
            ids=passages["id"].to_numpy(dtype=str),
            tokens=token_ids,
            lengths=lengths,
            **tables,
        )
        pending.replace(cache)
    print(f"Cached inputs in {cache}", flush=True)
    return cache


@dataclass(frozen=True)
class Run:
    name: str
    vectors: str = "glove50"
    hidden: int = 64
    head_hidden: tuple[int, ...] = ()
    dropout: float = 0.0
    learning_rate: float = 0.001
    mode: str = "frozen"
    fraction: float = 1.0
    readout: str = "last"
    word_dropout: float = 0.0


def settings(encoder: str) -> list[Run]:
    base = Run("glove50")
    runs = [
        base,
        Run("glove300", vectors="glove300"),
        Run("fasttext300", vectors="fasttext300"),
        replace(base, name="lr-3e-3", learning_rate=0.003),
        replace(base, name="finetuned", mode="finetuned"),
        replace(base, name="random", mode="random"),
    ]
    if encoder == "mean":
        runs.append(replace(base, name="mlp-head", head_hidden=(128,)))
    else:
        runs += [
            replace(base, name="hidden32", hidden=32),
            replace(base, name="train25", fraction=0.25),
            replace(base, name="train50", fraction=0.5),
        ]
    return runs


def pooling_grid() -> list[Run]:
    baselines = [
        Run(vectors, vectors=vectors) for vectors in ("glove50", "glove300", "fasttext300")
    ]
    models = []
    for vectors, hidden, rate, dropout in product(
        ("glove300", "fasttext300"), ((128,), (256, 128)), (0.001, 0.003, 0.01), (0.0, 0.25)
    ):
        layers = "x".join(map(str, hidden))
        models.append(
            Run(
                f"{vectors}-mlp{layers}-lr{rate:g}-drop{dropout:g}",
                vectors=vectors,
                head_hidden=hidden,
                learning_rate=rate,
                dropout=dropout,
            )
        )
    return baselines + models


def lstm_grid() -> list[Run]:
    return [
        Run(
            f"{vectors}-{readout}-lr{rate:g}",
            vectors=vectors,
            head_hidden=(128,),
            learning_rate=rate,
            readout=readout,
        )
        for vectors, readout, rate in product(
            ("glove300", "fasttext300"), ("last", "mean", "hybrid"), (0.001, 0.003)
        )
        if readout != "hybrid" or rate == 0.003
    ]


def lstm_followup(base: Run) -> list[Run]:
    return [
        base,
        replace(base, name="dropout", dropout=0.25),
        replace(base, name="word-dropout", word_dropout=0.15),
        replace(base, name="finetuned", mode="finetuned"),
        replace(base, name="random", mode="random"),
        replace(base, name="train25", fraction=0.25),
        replace(base, name="train50", fraction=0.5),
    ]


class SequenceClassifier(nn.Module):
    def __init__(self, encoder: str, config: Run, matrix: torch.Tensor):
        super().__init__()
        width = matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            matrix, freeze=config.mode == "frozen", padding_idx=0
        )
        self.lstm = nn.LSTM(width, config.hidden, batch_first=True) if encoder == "lstm" else None
        self.readout = config.readout
        self.word_dropout = config.word_dropout
        head_input = config.hidden if self.lstm is not None else width
        if self.lstm is not None and self.readout == "hybrid":
            head_input += width
        layers = []
        for units in config.head_hidden:
            layers.extend((nn.Linear(head_input, units), nn.ReLU(), nn.Dropout(config.dropout)))
            head_input = units
        layers.append(nn.Linear(head_input, len(LABELS)))
        self.head = nn.Sequential(*layers)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        vectors = self.embedding(ids)
        if self.training and self.word_dropout:
            keep = torch.rand(ids.shape, device=ids.device) >= self.word_dropout
            vectors = vectors * keep.unsqueeze(2)
        if self.lstm is None:
            features = vectors.sum(dim=1) / lengths.unsqueeze(1)
        else:
            sequence, _ = self.lstm(vectors)
            if self.readout == "last":
                ends = (lengths - 1).view(-1, 1, 1).expand(-1, 1, sequence.shape[-1])
                features = sequence.gather(1, ends).squeeze(1)
            else:
                positions = torch.arange(sequence.shape[1], device=ids.device)
                mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
                features = (sequence * mask.unsqueeze(2)).sum(dim=1) / lengths.unsqueeze(1)
                if self.readout == "hybrid":
                    features = torch.cat(
                        (F.normalize(features, dim=1), F.normalize(vectors.sum(dim=1), dim=1)),
                        dim=1,
                    )
        return F.normalize(features, dim=1)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(ids, lengths))


def batches(rows: np.ndarray, lengths: np.ndarray, size: int, rng=None):
    """Group lengths in 32-token bins so the LSTM spends less time on padding."""
    bins = (lengths[rows] + 31) // 32
    groups = []
    for bucket in np.unique(bins):
        selected = rows[bins == bucket]
        if rng is not None:
            selected = rng.permutation(selected)
        groups.extend((selected[i : i + size], bucket * 32) for i in range(0, len(selected), size))
    if rng is not None:
        rng.shuffle(groups)
    return groups


def score_fold(encoder, config, table, tokens, lengths, labels, train, val, args):
    # Use exactly the same training vocabulary for frozen, fine-tuned and random runs.
    used = np.unique(tokens[train])
    used = np.union1d(used, [0, 1])
    lookup = np.ones(len(table), dtype=np.int32)
    lookup[used] = np.arange(len(used))
    ids = torch.as_tensor(lookup[tokens], dtype=torch.long, device=args.device)
    lengths_device = torch.as_tensor(lengths, device=args.device)
    labels_device = torch.as_tensor(labels, device=args.device)
    matrix = torch.tensor(table[used])
    if config.mode == "random":
        rng = torch.Generator().manual_seed(args.seed + 1)
        matrix.normal_(std=matrix[2:].std().item(), generator=rng)
        matrix[:2].zero_()
    torch.manual_seed(args.seed)
    model = SequenceClassifier(encoder, config, matrix).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0001)
    rng = np.random.default_rng(args.seed)
    history = []
    stopper = TrainingLossStopper(
        DEFAULT_TRAINING.tol if args.early_stopping else 0.0, DEFAULT_TRAINING.patience
    )
    converged = False
    fixed_features = None
    if encoder == "mean" and config.mode == "frozen":
        fixed_features = matrix.new_empty((len(tokens), matrix.shape[1]), device=args.device)
        with torch.no_grad():
            for rows, width in batches(np.arange(len(tokens)), lengths, args.batch_size):
                index = torch.as_tensor(rows, device=args.device)
                fixed_features[index] = model.encode(ids[index, :width], lengths_device[index])

    def forward(rows, width):
        index = torch.as_tensor(rows, device=args.device)
        logits = (
            model(ids[index, :width], lengths_device[index])
            if fixed_features is None
            else model.head(fixed_features[index])
        )
        return logits, labels_device[index]

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = torch.zeros((), device=args.device)
        for rows, width in batches(train, lengths, args.batch_size, rng):
            opt.zero_grad(set_to_none=True)
            logits, targets = forward(rows, width)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            opt.step()
            train_loss += loss.detach() * len(rows)
        model.eval()
        with torch.no_grad():
            val_parts = [
                forward(rows, width) for rows, width in batches(val, lengths, args.batch_size)
            ]
            logits, targets = (torch.cat(items) for items in zip(*val_parts, strict=True))
            y = targets.cpu().numpy()
            predicted = logits.argmax(dim=1).cpu().numpy()
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss.item() / len(train),
                    "validation_loss": F.cross_entropy(logits, targets).item(),
                    "validation_accuracy": accuracy_score(y, predicted),
                    "validation_macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
                }
            )
        converged = stopper.should_stop(history[-1]["train_loss"])
        if converged:
            break
    return history, converged


def run(args, candidates: list[Run], group: str | None):
    start = perf_counter()
    cache = args.output / INPUT_CACHE
    if args.prepare_only or not cache.is_file():
        prepare(args.data_dir, args.output)
    if args.prepare_only:
        return
    passages = load_folds(args.data_dir).sort_values("id").reset_index(drop=True)
    with np.load(cache) as stored:
        if not np.array_equal(stored["ids"], passages["id"].to_numpy(dtype=str)):
            raise ValueError("Cached passage IDs differ from folds.csv. Run with --prepare-only.")
        tokens, lengths = stored["tokens"], stored["lengths"]
        tables = {name: stored[name] for name in MODELS}
    labels = passages["model"].map({name: index for index, name in enumerate(LABELS)}).to_numpy()
    folds = sorted(passages["fold"].unique().tolist())
    row_ids = {row: index for index, row in enumerate(passages["id"])}
    splits = {}
    for fold in folds:
        train = passages.loc[passages["fold"] != fold]
        val = passages.index[passages["fold"] == fold].to_numpy()
        if set(train["prompt_id"]) & set(passages.loc[val, "prompt_id"]):
            raise ValueError(f"Fold {fold} shares prompts between training and validation.")
        parts = assign_folds(train, n_folds=4, seed=DEFAULT_SEED)
        splits[fold] = {
            1.0: (train.index.to_numpy(), val),
            **{
                fraction: (parts.loc[parts["fold"] < groups, "id"].map(row_ids).to_numpy(), val)
                for fraction, groups in ((0.25, 1), (0.5, 2))
            },
        }
    output = args.output / "early-stopping" if args.early_stopping else args.output
    output = output / group / f"seed-{args.seed}" if group else output / args.encoder
    output.mkdir(parents=True, exist_ok=True)
    for config in candidates:
        if args.setting and config.name not in args.setting:
            continue
        result = {
            "encoder": args.encoder,
            "settings": asdict(config),
            "epochs": args.epochs,
            "early_stopping": (
                {
                    "train_loss_tolerance": DEFAULT_TRAINING.tol,
                    "patience": DEFAULT_TRAINING.patience,
                }
                if args.early_stopping
                else None
            ),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "tokenizer": WORD_TOKENIZER_NAME,
            "pretrained_model": MODELS[config.vectors],
            "device": str(args.device),
            "folds": [],
        }
        for fold in folds:
            train, val = splits[fold][config.fraction]
            history, converged = score_fold(
                args.encoder,
                config,
                tables[config.vectors],
                tokens,
                lengths,
                labels,
                train,
                val,
                args,
            )
            result["folds"].append(
                {
                    "fold": fold,
                    "train_rows": len(train),
                    "validation_rows": len(val),
                    "epochs_run": len(history),
                    "converged": converged,
                    "history": history,
                }
            )
            final = history[-1]
            print(
                f"{args.encoder} {config.name} fold {fold}: "
                f"accuracy {final['validation_accuracy']:.4f} "
                f"F1 {final['validation_macro_f1']:.4f} after {len(history)} epochs "
                f"({perf_counter() - start:.0f}s)",
                flush=True,
            )
            (output / f"{config.name}.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("artifacts/embeddings/fasttext-binary"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--encoder", choices=("mean", "lstm"), default="mean")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--pooling-grid", action="store_true")
    grid.add_argument("--lstm-grid", action="store_true")
    grid.add_argument("--lstm-reference", choices=[config.name for config in lstm_grid()])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Stop at the project training-loss threshold or the epoch cap.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--setting", action="append")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("Epochs and batch size must be positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Use --device cpu.")
    if args.pooling_grid and args.encoder != "mean":
        parser.error("--pooling-grid uses the mean encoder.")
    if (args.lstm_grid or args.lstm_reference) and args.encoder != "lstm":
        parser.error("LSTM studies require --encoder lstm.")
    group = None
    if args.pooling_grid:
        candidates, group = pooling_grid(), "pooling-mlp"
    elif args.lstm_grid:
        candidates, group = lstm_grid(), "lstm-grid"
    elif args.lstm_reference:
        reference = next(config for config in lstm_grid() if config.name == args.lstm_reference)
        candidates, group = lstm_followup(reference), f"lstm-followup/{reference.name}"
    else:
        candidates = settings(args.encoder)
    choices = {item.name for item in candidates}
    if args.setting and set(args.setting) - choices:
        parser.error(f"Choose settings from {', '.join(sorted(choices))}.")
    torch.set_num_threads(2)
    run(args, candidates, group)


if __name__ == "__main__":
    main()
