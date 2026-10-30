# Name That LLM

CPU baseline for [Kaggle's Name That LLM](https://www.kaggle.com/competitions/name-that-llm).
Identify the model that wrote a passage, using word and character TF-IDF features.

This project uses **prvsiyan's expanded forum dataset**, not the small official
training set. See the [dataset](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset)
and the author's [“Useful helper dataset” forum post](https://www.kaggle.com/prvsiyan/discussion).

## Files

```text
pyproject.toml, uv.lock       Python dependencies and exact resolved versions
scripts/download_data.py     Download and verify the expanded dataset
scripts/train_baseline.py    Prompt-grouped validation and final model training
scripts/predict.py           Write id/model predictions to Parquet
src/name_that_llm/           Reusable download, data, and model code
tests/                       Synthetic-data tests; no Kaggle data required
data/                        Local downloads (Git-ignored)
artifacts/                   Models, metrics, and predictions (Git-ignored)
```

This repository contains only the competition project. Downloads and generated
artifacts stay local and are excluded from version control.

## Quick start

Install [uv](https://docs.astral.sh/uv/) first if needed. `uv` manages the Python
3.12 environment.

```bash
git clone https://github.com/richardharnisch/name-that-llm.git
cd name-that-llm
uv sync --locked
uv run python scripts/download_data.py
uv run python scripts/train_baseline.py
```

If the dataset is already downloaded, the download command checks its file hashes
instead of downloading it again.

Training creates:

- `artifacts/baseline/model.joblib`: the pipeline refitted on all selected passages.
- `artifacts/baseline/metrics.json`: balanced accuracy, per-class results, settings,
  package versions, and the input file hash.
- `artifacts/baseline/oof_predictions.csv`: predictions from models that did not
  train on the corresponding prompt, with the prompt ID and fold number.

Existing nonempty result folders are not overwritten. For another experiment:

```bash
uv run python scripts/train_baseline.py --output-dir artifacts/experiment-02 --c 0.5
```

The equivalent installed commands are `uv run ntl-download`, `uv run ntl-train`,
and `uv run ntl-predict`. Use `--help` for their options.

### Verified local run

The default command was run on 2026-10-30 using dataset version 7: 4,818 passages,
282 prompt groups, five folds, and seed 42. Out-of-fold balanced accuracy was
**0.3065** (fold scores: 0.3105, 0.3141, 0.3060, 0.2896, 0.3124).
This is an external-data validation result, not a leaderboard result.
Trained artifacts are not included in this repository. Generate them with the
quick-start commands; use a new `--output-dir` for each later training run.

## Expanded data source

The downloader uses the public Kaggle dataset API, **not** the competition's
small `train.csv`. It does not read credentials, use API credits, or generate text.
Version **7** is pinned for reproducibility. Its actual files were checked on
2026-10-30: 12,548 total passages and 5,070 quality-filtered passages.

Downloaded files go to `data/raw/extended-v7/`. `download_manifest.json` records
the source, version, date, and SHA-256 hashes. The archive is checked for unsafe
paths before extraction. Downloaded Python files and notebooks are not executed.

The baseline reads `passages_clean.csv`, then selects these splits:

| Split | Passages | Prompt groups |
| --- | ---: | ---: |
| `new` | 4,158 | 231 |
| `new2` | 144 | 8 |
| `proxy_v1` | 516 | 43 |
| **Total** | **4,818** | **282** |

All 18 competition labels are represented. The baseline excludes `train_regen`
(reconstructed competition training prompts), `pilot_range` (a different length
protocol), and `new_kimi_nothink` (a Kimi-only rerun). It does not silently fall
back to the official small training set.

To use a different published version, download it into its own directory and
pass that directory to training. For example, to select the pinned version
explicitly:

```bash
uv run python scripts/download_data.py --version 7
uv run python scripts/train_baseline.py --data-dir data/raw/extended-v7 --output-dir artifacts/run-02
```

The dataset author reports differences from the competition's text distribution.
Check the dataset card, provider terms, and current competition rules before a
competition submission. Public availability does not by itself confirm that all
generation routes meet current rules. To exclude the `proxy_v1` split:

```bash
uv run python scripts/train_baseline.py --splits new new2 --output-dir artifacts/new-only
```

Some `new` passages also use the Kaggle proxy; the split selection above is not
a complete route filter. The source records each row's `provider_route`.

## Baseline and validation

- Word TF-IDF: one- and two-word features.
- Character TF-IDF: sequences of two to five characters, including punctuation.
- Preserve case and Unicode typography; use at most 100,000 features per view.
- Class-balanced linear SVM (`LinearSVC`); no GPU or model downloads needed.
- Five-fold `GroupKFold` by `prompt_id`, with seed 42. All passages for a prompt
  stay in the same fold. Vocabulary and IDF are fitted separately in each fold.
- Report **balanced accuracy**: the mean recall across the 18 models. Folds
  without all labels are rejected. Empty text, invalid labels, duplicate IDs,
  and duplicate passage text are also rejected.

Scores measure held-out prompts **within the external dataset**, not performance
on the competition test set. No leaderboard score is claimed. Do not use a
random passage-level split or treat synthetic test results as competition evidence.

## Predict locally

Download `test.csv` from the competition's **Data** tab after accepting its rules,
then put it at `data/raw/competition/test.csv`. No official data or credentials
are downloaded automatically.

```bash
uv run python scripts/predict.py
```

This writes `artifacts/submission.parquet` with exactly `id, model`. IDs remain
strings, row order is preserved, and duplicate IDs or invalid labels cause an
error. The output is not overwritten if it already exists. To select other paths:

```bash
uv run python scripts/predict.py --model artifacts/baseline/model.joblib --test-csv /path/to/test.csv --output artifacts/submission-02.parquet
```

Only load trusted `.joblib` files: loading one can execute Python code.

## Kaggle submission

This is a **code competition**. The local example test file is only a smoke test.
Kaggle must rerun a notebook on the hidden test set with internet disabled; a
locally generated sample submission is not a scored result.

Attach your trained `model.joblib` as a **private** Kaggle input. Use the same
scikit-learn, NumPy, and joblib versions used to train it; see `uv.lock` and
`metrics.json`. Cross-version loading is not supported. If the notebook's
environment differs, provide matching offline wheels or retrain there.
With one trusted `model.joblib` attached, the inference cell is:

```python
from pathlib import Path
import joblib
import pandas as pd

models = list(Path("/kaggle/input").rglob("model.joblib"))
assert len(models) == 1, "Attach exactly one baseline model."
model = joblib.load(models[0])
test = pd.read_csv(
    "/kaggle/input/competitions/name-that-llm/test.csv",
    dtype=str,
    keep_default_na=False,
)
assert test["id"].is_unique and test["text"].str.strip().ne("").all()
pd.DataFrame({"id": test["id"], "model": model.predict(test["text"])}).to_parquet(
    "/kaggle/working/submission.parquet", index=False
)
```

The local training and inference paths are tested. A Kaggle notebook submission
and hidden-test execution have **not** been verified.

## Checks

```bash
uv run pytest
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

Tests cover the public-download path using a synthetic ZIP, cached hash checks,
unsafe archive paths, split filtering, prompt separation, and the actual training
and prediction scripts. They do not need network access or competition data.

## Data credit and privacy

prvsiyan (2026). *prvsiyan's Extended Name That LLM Dataset*. Kaggle.
<https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset>

The dataset card lists CC BY 4.0 for generated passages. Reconstructed prompt
texts retain pjmathematician's CC BY-SA 4.0 terms; see the downloaded dataset
README for details. These are data licenses, not a license grant for this code.
Do not commit data, fitted models, text-derived outputs, submissions, or secrets.
