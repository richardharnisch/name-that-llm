"""Extract frozen, revision-pinned transformer features without author-label training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer

from name_that_llm.data import load_folds
from name_that_llm.research.run import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists.")
    if args.batch_size < 1 or args.max_length < 2:
        parser.error("Invalid batch size or sequence length.")
    torch.set_num_threads(4)
    frame = load_folds(args.data_dir)
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    cache = Path("models/huggingface")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, cache_dir=cache)
    model = AutoModel.from_pretrained(args.model, revision=revision, cache_dir=cache)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    vectors = []
    truncated = 0
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            texts = frame.text.iloc[start : start + args.batch_size].tolist()
            lengths = tokenizer(texts, truncation=False, return_length=True)["length"]
            truncated += sum(length > args.max_length for length in lengths)
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            vectors.append(pooled.cpu().numpy())
            if start % (args.batch_size * 20) == 0:
                print(f"{start}/{len(frame)} passages", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez_compressed(
            stream, ids=frame.id.to_numpy(dtype=str), embeddings=np.concatenate(vectors)
        )
    manifest = dict(
        model=args.model,
        revision=revision,
        pooling="masked mean, L2 normalized",
        max_length=args.max_length,
        truncated_rows=truncated,
        rows=len(frame),
        device=device,
        frozen=True,
        author_label_training=False,
        folds_sha256=digest(args.data_dir / "folds.csv"),
    )
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
