# Name That LLM

## Configuration

`src/name_that_llm/config.py` owns the experiment defaults in frozen dataclasses:

- `SplitConfig`: source groups, number of cross-validation folds `5`, and seed `42`.
- `TokenizerConfig`: BPE vocabulary size `8000` and minimum pair frequency `2`.
- `TrainingConfig`: model (`log-linear`, `mlp`, or `svm`), encoder (`bag-of-words`,
  `token-indicator`, or `tf-idf`), L2 normalization of the document vectors, the
  convergence rule (`max_epochs`, `tol`, `patience`), batch size, learning rate,
  weight decay, seed, and optional per-class row limit. BPE settings belong to
  `TokenizerConfig`.
- `MLPConfig`: hidden layer sizes `(256,)` and activation `ReLU`.

Paths are separate constants in the same file. Fixed labels remain in
`name_that_llm.data.LABELS`. Existing CLI flags override the defaults. For a separate
experiment configuration, use `dataclasses.replace` instead of changing shared state:

```python
from dataclasses import replace

from name_that_llm.config import DEFAULT_TOKENIZER
from name_that_llm.preprocess import preprocess_data

config = replace(DEFAULT_TOKENIZER, vocab_size=16000)
preprocess_data(vocab_size=config.vocab_size, min_frequency=config.min_frequency, fold=0)
```

Resolved settings are saved as `data/split_config.json`, `config.json` beside the
tokenized Parquet files of each fold, and `config.json` in each training output directory.
Split settings are for reproducibility, not parameters to optimize for higher scores.

## Prepare the data

```sh
uv run python scripts/download_data.py
uv run python scripts/preprocess_data.py
```

This downloads and verifies the public dataset, then creates:

- `data/folds.csv`: all eligible passages, with a `fold` column from `0` to `4`.
- `data/raw/extended-v7/`: unchanged source files.

`scripts/preprocess_data.py` then prepares the tokenizer and tokenized samples of every fold
(see "BPE preprocessing").

For an existing download, create the folds without downloading again:

```sh
uv run python scripts/split_data.py
```

### Cross-validation folds

The passages are divided into 5 folds. For fold `k`, the passages with `fold == k` are the
validation set, and the passages of the other four folds are the training set. Every passage
is a validation passage in exactly one fold.

- All passages with the same `prompt_id` stay in one fold, so a prompt never appears in both
  the training and validation set of one fold.
- The split balances the model labels across folds (scikit-learn `StratifiedGroupKFold`
  with shuffling and seed `42`). Every training and validation set must contain all
  18 model labels, or the split fails.
- The same dataset version, settings, and locked dependencies produce the same
  `folds.csv` on each machine, even if the source CSV row order changes. Output rows are
  sorted by ID and use UTF-8 with LF line endings. Repeat runs replace the file.
  Source files do not change.
- Only source groups `new`, `new2`, and `proxy_v1` are included. The CSV `split` column
  keeps these source group names.

To change the number of folds or the seed:

```sh
uv run ntl-split --folds 5 --seed 42
```

There is no separate held-out test set. All passages are used in the cross-validation, so
scores that you use to choose settings are optimistic estimates of performance on new data.

## Run a cross-validation experiment

From the project root, after preparing the data:

```sh
uv run ntl-train --model mlp --encoder bag-of-words
```

You can also use `uv run python scripts/train.py` or `uv run python -m name_that_llm.train`.

| Option | Choices |
|---|---|
| `--model` | `log-linear` (default), `mlp`, `svm` |
| `--encoder` | `bag-of-words` (default), `token-indicator`, `tf-idf` |

`svm` uses the same linear layer as `log-linear`, trained with a multi-class hinge loss.
Its scores are not probabilities, so its log loss is not reported.

For each fold, the command prepares or reuses the BPE tokenizer and tokenized samples, builds
the features from the saved `input_ids` without tokenizing again, trains a new model on the
training set of the fold, and evaluates it on the validation set of the fold. The tokenizer
and the TF-IDF weights are fitted on the training set of the fold only.

**Feature scale.** By default, each document vector is scaled to unit L2 norm
(`--normalize`; use `--no-normalize` to turn it off). The three encoders produce vectors
with very different sizes (bag-of-words about 50 times larger than TF-IDF), so one learning
rate cannot suit all of them without this step. The scaling has no fitted parameters.
Document length carries almost no label information in this dataset: the model label
explains less than 1% of the variation in token count.

**Convergence.** Training stops when the training loss has not beaten its best value so far
by at least the fraction `--tol` (default `0.01`) for `--patience` epochs (default `10`), or
at `--max-epochs` (default `200`; `--epochs` is an alias). In words: less than 1% training-loss
improvement over 10 epochs. The comparison is with the best loss, not with the previous
epoch, because the loss of each epoch is noisy (about 2% for the MLP once it plateaus).
Use `--tol 0` to always train for `--max-epochs`. The loss of a linear model on this
separable data never stops decreasing completely, so any threshold is a choice. This one was
fixed from training-loss curves only, before comparing validation scores. With it, all nine
model and encoder combinations stopped between epochs 117 and 149 on fold 0, below the cap. The validation data is evaluated each epoch for the learning curves only, and
never decides when to stop. The model of the last epoch is kept. Every run records whether it
converged. A run that hits the epoch cap did not converge, and its score is not comparable
with converged runs.

The script prints the scores of each fold and then the mean and sample standard deviation
over the folds for training accuracy, validation accuracy, macro F1, and log loss. The folds
share training data, so the standard deviation is a rough guide only. It saves these files to
`artifacts/simple_train/`:

- `cv_metrics.json`: scores of each fold, their mean and standard deviation, the epochs run
  and the converged flag of each fold, and the per-class report of the pooled out-of-fold
  predictions.
- `config.json`: resolved tokenizer and training settings, including CLI overrides.
- `confusion_matrix.png` and `per_class_f1.png`: made from the pooled out-of-fold
  predictions. Each passage is predicted once, by the model that did not train on it.
- `fold-K/model.pt`, `fold-K/metrics.json`, and `fold-K/learning_curves.png` for each fold.

To run only one fold, or to do a quick check:

```sh
uv run ntl-train --fold 0
uv run ntl-train --model svm --encoder token-indicator --limit-per-class 20 --output artifacts/quick
```

`--limit-per-class` samples at most that many rows per class from the training and
validation set of each fold after preprocessing. The tokenizer still uses the full training
set of the fold.

To change BPE settings and save a separate experiment:

```sh
uv run ntl-train --vocab-size 4000 --min-frequency 2 --output artifacts/bpe-4000
```

Use `--tokenizer-root` and `--tokenized-root` to select the cache roots, and `--data-dir`
to select the directory that contains `folds.csv`. That file needs `id`, `prompt_id`,
`text`, `model`, `split`, and `fold` columns, and all 18 labels. Use `--max-epochs`,
`--batch-size`, `--learning-rate`, `--weight-decay`, `--hidden`, `--seed`, and `--output`
to change the model settings and result directory. A saved `model.pt` records whether the
features were normalized and, for TF-IDF, the IDF weights, so a reloaded model builds the
same features. Checkpoints saved before this option existed load as not normalized.

## BPE preprocessing

`scripts/preprocess_data.py` reads `data/folds.csv`. For each fold, it fits a tokenizer on the
training set of that fold only, then tokenizes the training and validation sets once and
saves:

```text
artifacts/tokenizers/bpe-vs-8000-mf-2/fold-K/tokenizer.json
data/tokenized/bpe-vs-8000-mf-2/fold-K/train.parquet
data/tokenized/bpe-vs-8000-mf-2/fold-K/val.parquet
data/tokenized/bpe-vs-8000-mf-2/fold-K/preprocessing_manifest.json
```

A separate tokenizer for each fold is necessary. A tokenizer fitted on all passages would
learn its vocabulary and merge rules from the validation passages of every fold.

This follows `../nlp1-student/part1/tutorial.ipynb`: Hugging Face `Tokenizer(BPE(...))`,
`Metaspace` pretokenization, and `[UNK]`. The default vocabulary size is 8,000,
with minimum pair frequency 2.
There is no lowercasing or additional Unicode normalization. Unseen characters map to
`[UNK]`. Samples are not padded or truncated, and no beginning/end tokens are added.

Vocabulary size remains a hyperparameter to tune later using cross-validated accuracy
and macro F1. Compare 1,000, 4,000, 8,000, and 16,000; 8,000 is a starting
point, not an established optimum.

Each Parquet row retains the source columns and adds `input_ids` (a list of integer
token IDs) and `label` (the model's index in `name_that_llm.data.LABELS`). The manifest
records settings, labels, library versions, row counts, and file hashes. Repeat runs
reuse the saved files if the inputs, settings, library versions, and output hashes match.
Otherwise, preprocessing rebuilds the generated files of that fold. Raw text and
`folds.csv` do not change.

To use different settings, or to process one fold only:

```sh
uv run ntl-preprocess --vocab-size 4000 --min-frequency 2
uv run ntl-preprocess --fold 0
```

The encoders support both the tutorial's text interface and saved token IDs:

```python
import pandas as pd

from name_that_llm.encodings import BagOfWords, TokenIndicator
from name_that_llm.tokenizer import Tokenizer

tokenizer = Tokenizer.from_file("artifacts/tokenizers/bpe-vs-8000-mf-2/fold-0/tokenizer.json")
train = pd.read_parquet("data/tokenized/bpe-vs-8000-mf-2/fold-0/train.parquet")
bow = BagOfWords(tokenizer)
indicator = TokenIndicator(tokenizer)

features = bow.from_ids(train.iloc[0]["input_ids"])  # No repeated tokenization.
indicators = indicator.from_ids(train.iloc[0]["input_ids"])
prediction_features = bow("An unseen passage.")  # The original text interface still works.
```
