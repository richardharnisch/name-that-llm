# Name That LLM

## Prepare the data

```sh
uv run python scripts/download_data.py
uv run python scripts/preprocess_data.py
```

This downloads and verifies the public dataset, then creates:

- `data/train.csv`: training passages.
- `data/val.csv`: validation passages.
- `data/raw/extended-v7/`: unchanged source files.

For an existing download, create the split without downloading again:

```sh
uv run python scripts/split_data.py
```

The default split uses 80% of prompts for training and 20% for validation, with
seed `42`. All passages with the same `prompt_id` stay in one set. The row ratio
can differ slightly because prompts have different numbers of passages. Both
sets must contain all 18 model labels.

The same dataset version, settings, and locked dependencies produce the same
files on each machine, even if the source CSV row order changes. Output rows are
sorted by ID and use UTF-8 with LF line endings. Repeat runs replace the generated
CSV files. Source files do not change.

Only source groups `new`, `new2`, and `proxy_v1` are included. The CSV `split`
column keeps these source group names; the file name identifies train or validation.
Use only `train.csv` to fit models and preprocessing; use `val.csv` for evaluation.

To change the validation fraction or seed:

```sh
uv run ntl-split --val-size 0.2 --seed 42
```

## BPE preprocessing

`scripts/preprocess_data.py` reads the existing `data/train.csv` and `data/val.csv`.
It fits the tokenizer on training text only, then tokenizes both sets once and saves:

```text
artifacts/tokenizers/bpe-vs-8000-mf-2/tokenizer.json
data/tokenized/bpe-vs-8000-mf-2/train.parquet
data/tokenized/bpe-vs-8000-mf-2/val.parquet
data/tokenized/bpe-vs-8000-mf-2/preprocessing_manifest.json
```

This follows `../nlp1-student/part1/tutorial.ipynb`: Hugging Face `Tokenizer(BPE(...))`,
`Metaspace` pretokenization, and `[UNK]`. The default vocabulary size is 8,000,
with minimum pair frequency 2.
There is no lowercasing or additional Unicode normalization. Unseen characters map to
`[UNK]`. Samples are not padded or truncated, and no beginning/end tokens are added.

Vocabulary size remains a hyperparameter to tune later using classifier validation
accuracy and macro-F1. Compare 1,000, 4,000, 8,000, and 16,000; 8,000 is a starting
point, not an established optimum.

Each Parquet row retains the source columns and adds `input_ids` (a list of integer
token IDs) and `label` (the model's index in `name_that_llm.data.LABELS`). The manifest
records settings, labels, library versions, row counts, and file hashes. Repeat runs
reuse the saved files if the inputs, settings, library versions, and output hashes match.
Otherwise, preprocessing rebuilds the generated files. Raw text and split CSV files
do not change.

To use different settings:

```sh
uv run ntl-preprocess --vocab-size 4000 --min-frequency 2
```

The encoders support both the tutorial's text interface and saved token IDs:

```python
import pandas as pd

from name_that_llm.encodings import BagOfWords, TokenIndicator
from name_that_llm.tokenizer import Tokenizer

tokenizer = Tokenizer.from_file("artifacts/tokenizers/bpe-vs-8000-mf-2/tokenizer.json")
train = pd.read_parquet("data/tokenized/bpe-vs-8000-mf-2/train.parquet")
bow = BagOfWords(tokenizer)
indicator = TokenIndicator(tokenizer)

features = bow.from_ids(train.iloc[0]["input_ids"])  # No repeated tokenization.
indicators = indicator.from_ids(train.iloc[0]["input_ids"])
prediction_features = bow("An unseen passage.")  # The original text interface still works.
```
