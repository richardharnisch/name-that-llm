"""Fine-tune a small pretrained transformer to name the LLM that wrote a passage.

Uses the prompt-grouped split from prepare.py. Runs on Apple MPS, CUDA or CPU.
"""

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def batches(ids, labels, size, order, tokenizer, device):
    for start in range(0, len(order), size):
        index = order[start : start + size]
        batch = tokenizer.pad({"input_ids": [ids[i] for i in index]}, return_tensors="pt")
        yield {k: v.to(device) for k, v in batch.items()}, torch.tensor(labels[index]).to(device)


@torch.no_grad()
def predict(model, ids, labels, args, tokenizer, device):
    model.eval()
    order = np.argsort([len(x) for x in ids])  # sort by length so batches pad little
    out = np.zeros(len(ids), dtype=int)
    position = 0
    for batch, _ in batches(ids, labels, args.eval_batch_size, order, tokenizer, device):
        out[order[position : position + len(batch["input_ids"])]] = model(**batch).logits.argmax(-1).cpu().numpy()
        position += len(batch["input_ids"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--train-size", type=int, help="Use a random subset of the training set")
    parser.add_argument("--eval-size", type=int, default=5000, help="Rows of val/test to score")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=float, default=7.0, help="Stop training after this long")
    parser.add_argument("--save-model", help="Save the fine-tuned weights here")
    parser.add_argument("--output", default="results_transformer.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    frame = pd.read_parquet("data/pool.parquet")
    labels_names = sorted(frame["model"].unique())
    lookup = {name: i for i, name in enumerate(labels_names)}
    parts = {s: frame[frame["split"] == s] for s in ("train", "val", "test")}
    if args.train_size:
        parts["train"] = parts["train"].sample(args.train_size, random_state=args.seed)
    for s in ("val", "test"):
        parts[s] = parts[s].sample(min(args.eval_size, len(parts[s])), random_state=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids, labels = {}, {}
    for s, part in parts.items():
        ids[s] = tokenizer(part["text"].tolist(), truncation=True, max_length=args.max_length)["input_ids"]
        labels[s] = part["model"].map(lookup).to_numpy()
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels_names)).to(device)

    steps = args.epochs * int(np.ceil(len(ids["train"]) / args.batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    rng = np.random.default_rng(args.seed)
    start, step, out_of_time = perf_counter(), 0, False
    epoch_scores = []
    for epoch in range(args.epochs):
        model.train()
        # Shuffle, then sort within chunks of 50 batches so batches have similar lengths.
        order = rng.permutation(len(ids["train"]))
        chunk = args.batch_size * 50
        order = np.concatenate([c[np.argsort([len(ids["train"][i]) for i in c])] for c in np.array_split(order, max(1, len(order) // chunk))])
        batch_order = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        rng.shuffle(batch_order)
        for index in batch_order:
            batch, target = next(batches(ids["train"], labels["train"], len(index), index, tokenizer, device))
            loss = torch.nn.functional.cross_entropy(model(**batch).logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            step += 1
            if perf_counter() - start > args.max_hours * 3600:
                print("Time limit reached; stopping training early.", flush=True)
                out_of_time = True
                break
            if step % 50 == 0:
                rate = step * args.batch_size / (perf_counter() - start)
                print(f"step {step}/{steps} loss {loss.item():.3f} {rate:.1f} samples/s", flush=True)

        pred = predict(model, ids["val"], labels["val"], args, tokenizer, device)
        epoch_scores.append({"epoch": epoch + 1, "val_acc": accuracy_score(labels["val"], pred)})
        print(f"epoch {epoch + 1} val acc {epoch_scores[-1]['val_acc']:.4f}", flush=True)
        if out_of_time:
            break

    train_seconds = perf_counter() - start
    result = {"model": args.model, "train_rows": len(ids["train"]), "epochs": args.epochs,
              "epoch_scores": epoch_scores, "stopped_for_time": out_of_time, "train_seconds": train_seconds, "samples_per_second": len(ids["train"]) * args.epochs / train_seconds}
    for s in ("val", "test"):
        pred = predict(model, ids[s], labels[s], args, tokenizer, device)
        result[f"{s}_acc"] = accuracy_score(labels[s], pred)
        result[f"{s}_macro_f1"] = f1_score(labels[s], pred, average="macro")
    print(result)
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
    json.dump(result, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
