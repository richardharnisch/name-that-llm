import pandas as pd
import pytest

from name_that_llm.data import LABELS, load_test_data, load_training_data, validate_columns


def test_load_expanded_data(data_dir, training_frame):
    loaded = load_training_data(data_dir)
    pd.testing.assert_frame_equal(loaded, training_frame[loaded.columns])


@pytest.mark.parametrize("split", ["train_regen", "pilot_range", "new_kimi_nothink"])
def test_excludes_unsafe_splits(data_dir, training_frame, split):
    excluded = training_frame.iloc[[0]].assign(id="excluded", prompt_id="excluded", split=split)
    pd.concat([training_frame, excluded]).to_csv(data_dir / "passages_clean.csv", index=False)
    loaded = load_training_data(data_dir)
    assert len(loaded) == len(training_frame)
    assert "excluded" not in set(loaded["id"])
    with pytest.raises(ValueError, match="training splits"):
        load_training_data(data_dir, (split,))


def test_missing_download_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="download_data.py"):
        load_training_data(tmp_path)


def test_rejects_missing_labels(data_dir, training_frame):
    training_frame.loc[training_frame["model"] != LABELS[0]].to_csv(
        data_dir / "passages_clean.csv", index=False
    )
    with pytest.raises(ValueError, match="18 competition labels"):
        load_training_data(data_dir)


def test_rejects_unknown_labels(data_dir, training_frame):
    training_frame.loc[0, "model"] = "not-a-competition-label"
    training_frame.to_csv(data_dir / "passages_clean.csv", index=False)
    with pytest.raises(ValueError, match="unknown"):
        load_training_data(data_dir)


def test_rejects_whitespace_normalised_duplicates(data_dir, training_frame):
    training_frame.loc[1, "text"] = "  " + training_frame.loc[0, "text"] + "\n"
    training_frame.to_csv(data_dir / "passages_clean.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate passage"):
        load_training_data(data_dir)


@pytest.mark.parametrize("value", ["", "   ", None])
def test_rejects_empty_text(value):
    with pytest.raises(ValueError, match="empty values"):
        validate_columns(pd.DataFrame({"text": [value]}), ("text",))


def test_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate IDs"):
        validate_columns(pd.DataFrame({"id": ["same", "same"]}), ("id",))


def test_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(pd.DataFrame({"id": ["1"]}), ("id", "text"))


def test_test_ids_remain_strings(tmp_path):
    path = tmp_path / "test.csv"
    pd.DataFrame({"id": ["00123", "NA"], "text": ["First text.", "Second text."]}).to_csv(
        path, index=False
    )
    assert load_test_data(path)["id"].tolist() == ["00123", "NA"]
