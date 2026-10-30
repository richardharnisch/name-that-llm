import pandas as pd
import pytest

from name_that_llm.data import LABELS


@pytest.fixture
def training_frame():
    """Synthetic passages only: tests do not require or publish Kaggle data."""
    return pd.DataFrame(
        [
            {
                "id": f"prompt{prompt:03}__{label}",
                "prompt_id": f"prompt{prompt:03}",
                "model": label,
                "text": f"Synthetic example {prompt} with distinctive style {label}.",
                "split": "new",
            }
            for prompt in range(6)
            for label in LABELS
        ]
    )


@pytest.fixture
def data_dir(tmp_path, training_frame):
    directory = tmp_path / "data"
    directory.mkdir()
    training_frame.to_csv(directory / "passages_clean.csv", index=False)
    return directory
