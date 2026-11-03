from pathlib import Path

import pandas as pd
import pytest

from name_that_llm.data import LABELS, split_training_data, write_data_splits


def source_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": f"{prompt:03d}__{model}",
                "prompt_id": f"{prompt:03d}",
                "text": f"Passage {prompt} by {model}",
                "model": model,
                "split": "new",
            }
            for prompt in range(10)
            for model in LABELS
        ]
    )


def test_split_is_deterministic_and_keeps_prompts_together() -> None:
    frame = source_data()
    train, val = split_training_data(frame)
    shuffled_train, shuffled_val = split_training_data(frame.sample(frac=1, random_state=7))

    pd.testing.assert_frame_equal(train, shuffled_train)
    pd.testing.assert_frame_equal(val, shuffled_val)
    assert train["prompt_id"].nunique() == 8
    assert val["prompt_id"].nunique() == 2
    assert set(train["prompt_id"]).isdisjoint(val["prompt_id"])
    assert set(train["id"]).isdisjoint(val["id"])
    assert set(train["id"]) | set(val["id"]) == set(frame["id"])
    assert set(train["model"]) == set(val["model"]) == set(LABELS)
    _, other_val = split_training_data(frame, seed=7)
    assert set(val["id"]) != set(other_val["id"])


def test_split_rejects_invalid_inputs() -> None:
    frame = source_data()
    for val_size in (0, 1, -0.1, 1.1):
        with pytest.raises(ValueError, match="Validation size"):
            split_training_data(frame, val_size=val_size)
    with pytest.raises(ValueError, match="At least two distinct prompts"):
        split_training_data(frame.loc[frame["prompt_id"] == "000"])
    frame.loc[frame["model"] == LABELS[0], "prompt_id"] = "rare"
    with pytest.raises(ValueError, match="every model"):
        split_training_data(frame)


def test_write_data_splits_filters_sources_and_repeats_exactly(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    frame = source_data()
    excluded = frame.iloc[[0]].assign(id="excluded", text="Excluded passage", split="train_regen")
    path = source / "passages_clean.csv"
    pd.concat([frame, excluded]).to_csv(path, index=False)
    original = path.read_bytes()
    output = tmp_path / "prepared"

    write_data_splits(source, output)
    first = {name: (output / name).read_bytes() for name in ("train.csv", "val.csv")}
    write_data_splits(source, output)
    assert first == {name: (output / name).read_bytes() for name in first}
    assert path.read_bytes() == original
    for name in first:
        subset = pd.read_csv(output / name, dtype=str, keep_default_na=False)
        assert set(subset["split"]) == {"new"}
        assert subset["prompt_id"].str.len().eq(3).all()
        assert "excluded" not in set(subset["id"])
