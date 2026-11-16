"""No-download checks for supervised encoder training."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from name_that_llm.data import LABELS, split_fold
from name_that_llm.research.finetune import EncoderClassifier, encode, train_model


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6)
        self.embedding = nn.Embedding(64, 6)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyTokenizer:
    def __init__(self):
        self.seen = []

    def __call__(self, texts, **kwargs):
        self.seen.extend(texts)
        ids = torch.tensor([[int(text), 0] for text in texts])
        return {"input_ids": ids, "attention_mask": torch.tensor([[1, 0]] * len(ids))}


def rows():
    return pd.DataFrame(
        [
            {
                "id": f"{fold}-{i}",
                "text": str(i + 1),
                "model": label,
                "prompt_id": f"p{fold}",
                "fold": fold,
            }
            for fold in (0, 1)
            for i, label in enumerate(LABELS)
        ]
    )


def test_only_training_partition_updates_encoder():
    torch.manual_seed(42)
    model = EncoderClassifier(TinyEncoder())
    tokenizer = TinyTokenizer()
    frame = rows()
    # Held-out tokens are different, so any label leakage or encoding is visible.
    frame.loc[frame.fold == 0, "text"] = "31"
    train, validation = split_fold(frame, 0)
    before = model.encoder.embedding.weight.detach().clone()
    completed, steps = train_model(
        model,
        tokenizer,
        train,
        device="cpu",
        epochs=2,
        batch_size=18,
        learning_rate=0.01,
        weight_decay=0,
    )
    assert completed == steps == 2
    assert "31" not in tokenizer.seen
    assert not torch.equal(before, model.encoder.embedding.weight)
    assert torch.equal(before[31], model.encoder.embedding.weight[31])
    assert set(train.id).isdisjoint(validation.id)
    vectors, probabilities = encode(
        model, tokenizer, frame, device="cpu", batch_size=12, max_length=8
    )
    assert vectors.shape == (36, 6)
    assert probabilities.shape == (36, 18)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)


def test_step_limit_reports_incomplete_epochs():
    model = EncoderClassifier(TinyEncoder())
    train, _ = split_fold(rows(), 0)
    completed, steps = train_model(
        model, TinyTokenizer(), train, device="cpu", epochs=3, batch_size=4, max_steps=1
    )
    assert (completed, steps) == (0, 1)


def test_pooling_ignores_padding():
    model = EncoderClassifier(TinyEncoder())
    features = model.embed(
        {
            "input_ids": torch.tensor([[1, 2], [1, 9]]),
            "attention_mask": torch.tensor([[1, 0], [1, 0]]),
        }
    )
    torch.testing.assert_close(features[0], features[1])


def test_cli_exports_complete_and_rejects_overwrite(tmp_path, monkeypatch):
    import json
    import sys

    import pytest

    from name_that_llm.research import finetune
    from name_that_llm.research.run import digest

    class LocalTokenizer(TinyTokenizer):
        def __call__(self, texts, **kwargs):
            if kwargs.get("return_length"):
                return {"length": [1] * len(texts)}
            return super().__call__(texts, **kwargs)

    class LocalEncoder(TinyEncoder):
        def __init__(self):
            super().__init__()
            self.config.to_dict = lambda: {"hidden_size": 6}

    frame = rows()
    frame["split"] = "train"
    # Preserve unique text, with enough token vocabulary for both folds.
    frame.loc[frame.fold == 0, "text"] = [str(i + 1) for i in range(18)]
    frame.loc[frame.fold == 1, "text"] = [str(i + 19) for i in range(18)]
    frame.to_csv(tmp_path / "folds.csv", index=False)
    monkeypatch.setattr(finetune.AutoTokenizer, "from_pretrained", lambda *a, **k: LocalTokenizer())
    monkeypatch.setattr(finetune.AutoModel, "from_pretrained", lambda *a, **k: LocalEncoder())
    output = tmp_path / "artifact"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune",
            "--data-dir",
            str(tmp_path),
            "--fold",
            "0",
            "--output",
            str(output),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ],
    )
    finetune.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_training"] is True
    assert manifest["epochs_completed"] == 1
    assert set(manifest["train_ids"]).isdisjoint(manifest["validation_ids"])
    assert manifest["embeddings_sha256"] == digest(output / "embeddings.npz")
    with np.load(output / "embeddings.npz", allow_pickle=False) as archive:
        assert archive["ids"].dtype.kind == "U"
        assert len(archive["embeddings"]) == len(frame)
    predictions = pd.read_parquet(output / "classifier_predictions.parquet")
    assert set(predictions.id) == set(manifest["validation_ids"])
    assert set(predictions.method) == {"finetuned_head"}
    with pytest.raises(SystemExit):
        finetune.main()
