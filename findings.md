# Findings

## What the assignment asks, and how we address it

The checklist in `nlp1-student/part1/project.ipynb` has four requirements.

| Requirement | How we address it | Where |
|---|---|---|
| Compare text classification systems on our data and task | 9 systems (3 models x 3 encoders) on 18-way LLM attribution | Section 1 |
| Cover the two tutorial representations plus one of our own | Token indicators and bag-of-words (tutorial), plus **tf-idf** | Section 1 |
| Cover the two tutorial architectures plus one of our own | Log-linear and one-hidden-layer MLP (tutorial), plus a **linear SVM** (same linear layer, trained with a multi-class hinge loss) | Section 1 |
| Cross-validation and a relevant metric | 5-fold CV with whole prompts kept in one fold. Validation accuracy is the main metric, with macro F1 and log loss also recorded | Setup |
| Bayesian estimation with a ROPE, with a size we motivate | Bayesian correlated t-test on per-fold accuracy differences, ROPE of +-0.01 | Section 2 |

Still open: the SVM is a modification of the log-linear model, not of the MLP, so a deeper or dropout MLP variant would be a closer match to the notebook's suggestion (see "Not done").

## Setup

- Task: 18-class classification of which LLM wrote a passage (chance is about 5.6%).
- Data: 4,818 passages from 282 prompts, in 5 cross-validation folds. Whole prompts stay in one fold, so validation prompts are unseen.
- Tokenizer: BPE with a vocabulary of 8000 and minimum frequency 2, trained per fold on the training passages only.
- Models: log-linear, MLP (one hidden layer of 256 units) and linear SVM.
- Encoders: bag-of-words, token-indicator and tf-idf, all L2-normalised. The tf-idf weights come from the training passages only.
- Training: Adam, learning rate 1e-3, weight decay 1e-4, batch size 64. Training stops when the training loss improves by less than 1% over 10 epochs (cap 200). Validation data never decides when to stop.
- Comparison: Bayesian correlated t-test with a ROPE (`src/name_that_llm/results.py`), on per-fold validation accuracy.

## 1. Baseline grid (3 models x 3 encoders)

Train accuracy, validation accuracy and validation macro F1, mean over 5 folds (± is the std over folds). Runs are in `artifacts/rope/`.

| Encoder | Model | Train acc | Val acc | Val macro F1 |
|---|---|---|---|---|
| bag-of-words | log-linear | 0.887 | 0.294 ± 0.027 | 0.263 ± 0.025 |
| bag-of-words | MLP | 1.000 | 0.297 ± 0.021 | **0.285 ± 0.021** |
| bag-of-words | SVM | 0.919 | 0.291 ± 0.020 | 0.274 ± 0.021 |
| token-indicator | log-linear | 0.983 | **0.304 ± 0.027** | 0.282 ± 0.025 |
| token-indicator | MLP | 1.000 | 0.281 ± 0.020 | 0.271 ± 0.019 |
| token-indicator | SVM | 0.990 | 0.292 ± 0.022 | 0.279 ± 0.019 |
| tf-idf | log-linear | 0.998 | 0.284 ± 0.018 | 0.261 ± 0.018 |
| tf-idf | MLP | 1.000 | 0.269 ± 0.021 | 0.261 ± 0.022 |
| tf-idf | SVM | 0.998 | 0.274 ± 0.018 | 0.263 ± 0.019 |

- Every run scores 27-30%, far above chance but low in absolute terms.
- The models memorise the training data: train accuracy is 0.89-1.00, and the gap to validation is about 0.6-0.7. The gap is largest for tf-idf and the MLP, which reach almost 1.0 on the training data.
- Differences between runs are mostly within the fold-to-fold noise (std about 0.02).

## 2. ROPE analysis

### Why this is Bayesian

A cross-validation mean with a standard deviation cannot tell us whether two systems differ, and a classical significance test answers the wrong question. A frequentist t-test gives a p-value under the assumption that the two systems are exactly equal. It can reject that assumption, but it can never tell us that two systems are practically the same, because failing to reject is not evidence of equality. With only 5 folds it also has little power.

The Bayesian approach (Benavoli et al., 2017) instead puts a posterior distribution on the true mean accuracy difference between two systems A and B, given the per-fold differences. From that posterior we can read off three probabilities directly:

- **P(A better):** the probability that A beats B by more than r.
- **P(ROPE):** the probability that the difference lies in [-r, r], so the systems are practically equivalent.
- **P(B better):** the probability that B beats A by more than r.

This is the statement we actually want, and "practically equivalent" becomes something we can state with a probability rather than an absence of evidence.

### Method

- **ROPE (region of practical equivalence):** the interval [-r, r] of accuracy differences we treat as "the same model".
- **Paired differences:** for each pair of systems we take the per-fold difference in validation accuracy. All systems use the same 5 folds, so the pairing is valid.
- **Correlated t-test:** the folds share training passages, so their scores are correlated. The posterior is a Student t with n-1 degrees of freedom, centred on the mean difference, with scale sqrt(var * (1/n + rho/(1-rho))) and rho = 1/n (the Nadeau and Bengio correction). Ignoring rho would make the posterior too narrow and overstate how sure we are.
- **Choice of r = 0.01 (one accuracy point):** this is a choice, not a derived value. One point is about 48 of the 4,818 passages, which is a difference we would not call meaningful. It is also smaller than the fold-to-fold std of about 0.02, so it keeps the analysis from declaring everything equivalent. We use r = 0.05 as a sensitivity check.

Output is saved in `artifacts/rope/rope_acc.json` (r = 0.01). Command: `uv run python scripts/results.py artifacts/rope/<run> ... --rope <r>`.

### Results

- **r = 0.05:** all 36 pairs have P(ROPE) of at least 0.89. No model is distinguishable from any other. The largest mean gap is 3.5 points, so this ROPE is too wide to be informative.
- **r = 0.01:** most pairs are unresolved (P(ROPE) between 0.4 and 0.7, with a real chance of either model being better).

Selected pairs at r = 0.01 (A − B is the mean per-fold accuracy difference; P(A>B) means A is better by more than r):

| A | B | A − B | P(A>B) | P(ROPE) | P(B>A) |
|---|---|---|---|---|---|
| log-linear + token-indicator | MLP + tf-idf | +0.035 | 0.976 | 0.021 | 0.004 |
| log-linear + bag-of-words | MLP + tf-idf | +0.025 | 0.975 | 0.024 | 0.001 |
| MLP + bag-of-words | MLP + tf-idf | +0.027 | 0.942 | 0.052 | 0.006 |
| log-linear + token-indicator | MLP + token-indicator | +0.023 | 0.930 | 0.065 | 0.005 |
| SVM + bag-of-words | MLP + tf-idf | +0.021 | 0.904 | 0.089 | 0.006 |
| MLP + bag-of-words | SVM + token-indicator | +0.005 | 0.040 | 0.959 | 0.001 |
| SVM + bag-of-words | SVM + token-indicator | −0.001 | 0.114 | 0.739 | 0.147 |
| log-linear + bag-of-words | MLP + bag-of-words | −0.003 | 0.115 | 0.656 | 0.228 |
| log-linear + bag-of-words | log-linear + token-indicator | −0.011 | 0.014 | 0.453 | 0.532 |
| log-linear + token-indicator | MLP + bag-of-words | +0.008 | 0.379 | 0.597 | 0.024 |

The first five rows are the clearest differences, the next two are probably equivalent, and the last three are unresolved. All 36 pairs are in `artifacts/rope/rope_acc.json`.

- Clearest result at r = 0.01: MLP + tf-idf (0.269) is probably worse than log-linear + bag-of-words (P = 0.975), log-linear + token-indicator (P = 0.976) and MLP + bag-of-words (P = 0.942).
- No model is clearly the best. Log-linear + token-indicator has the highest mean, but its probability of beating most others is only 0.6-0.9.
- Conclusion: representation and architecture matter little here, apart from tf-idf with the MLP.

## 3. Regularization and capacity (small data)

All runs here use the small data. None of this was run on the big data.

### 3a. L2 regularization (weight decay)

The regularizer is L2: `torch.optim.Adam(..., weight_decay=λ)` adds λ·w to every weight's gradient, which is the gradient of an L2 penalty. It is plain Adam, not AdamW, and biases are penalised too. There is no L1 term anywhere. Weight decay 1e-3, 1e-2 and 1e-1 was tried on three setups (the baseline is 1e-4). Runs are in `artifacts/wd/`.

| Setup | Weight decay | Train acc | Val acc |
|---|---|---|---|
| Log-linear + bag-of-words | 1e-4 (baseline) | 0.887 | 0.294 |
| | 1e-3 | 0.489 | 0.243 |
| | 1e-2 | 0.274 | 0.184 |
| | 1e-1 | 0.163 | 0.126 |
| Log-linear + token-indicator | 1e-4 (baseline) | 0.983 | 0.304 |
| | 1e-3 | 0.686 | 0.281 |
| | 1e-2 | 0.497 | 0.231 |
| | 1e-1 | 0.216 | 0.125 |
| MLP + bag-of-words | 1e-4 (baseline) | 1.000 | 0.297 |
| | 1e-3 | 0.997 | 0.304 |
| | 1e-2 | 0.113 | 0.112 |
| | 1e-1 | 0.059 | 0.059 |

- Stronger L2 did not improve validation accuracy. The only setting level with or above the baseline is the MLP at 1e-3 (0.304 against 0.297, inside the noise).
- The train/validation gap shrinks only because train accuracy falls. The models underfit.
- The MLP collapses to near chance at 1e-2 and above.
- Early stopping is not the cause. Rerunning 1e-3 and 1e-2 for a fixed 100 epochs with no early stopping (`artifacts/wd_fixed/`) changed train and validation accuracy by at most about 0.06, and no run improved.

### 3b. Fewer parameters (MLP hidden size)

The MLP is 8000 -> h -> 18: one hidden layer of h ReLU units between the 8,000-dimensional input and the 18 output logits, with biases, no dropout and no normalisation. The baseline is h = 256. Only the width h was varied (bag-of-words, small data, same training settings). Runs are in `artifacts/hidden/`. Parameter counts are computed (8000·h + h·18 plus biases), not logged.

| Hidden units | Parameters | Train acc | Val acc | Val macro F1 |
|---|---|---|---|---|
| 256 (baseline) | 2.06M | 1.000 ± 0.000 | 0.297 ± 0.021 | 0.285 ± 0.021 |
| 128 | 1.03M | 1.000 ± 0.000 | 0.296 ± 0.020 | 0.286 ± 0.020 |
| 64 | 0.51M | 1.000 ± 0.000 | 0.290 ± 0.022 | 0.281 ± 0.022 |
| 32 | 0.26M | 1.000 ± 0.000 | 0.292 ± 0.023 | 0.283 ± 0.023 |
| 16 | 0.13M | 1.000 ± 0.000 | 0.290 ± 0.021 | 0.281 ± 0.019 |
| 8 | 64k | 1.000 ± 0.000 | 0.253 ± 0.018 | 0.249 ± 0.016 |
| 4 | 32k | 0.921 ± 0.064 | 0.211 ± 0.012 | 0.206 ± 0.012 |
| 2 | 16k | 0.657 ± 0.025 | 0.168 ± 0.014 | 0.155 ± 0.013 |

- From 256 down to 16 units, validation accuracy stays at 0.29-0.30 while train accuracy stays at 1.000. Capacity is not the lever.
- Below 16 the model starts to lose validation accuracy, and from 4 units down it also stops memorising. The gap shrinks only because the model gets worse.
- Same pattern as 3a: no setting reduces the train/validation gap while keeping the validation score.
- Only bag-of-words was swept (the other encoders were not), and the token-indicator sweep was cancelled for time.

## 4. Big data approach (Open Pool, 212k passages)

Code, data and logs are in `big-data-approach/` (see its `README.md`). The data is crtproduction's [Name That LLM Open Pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool): 212,623 passages from the same 18 models, median 102 words, 35,573 prompt groups. This is a different and about 44 times larger dataset than the 4,818 passages in sections 1-3.

### Setup

The test is the same as in the small-data runs: log-linear, MLP and SVM on the token-indicator encoder (the best encoder in section 1), 5-fold cross-validation with whole prompts in one fold, a BPE tokenizer (vocabulary 8000) fitted per fold, and the same hyperparameters and stopping rule (Adam, learning rate 1e-3, weight decay 1e-4, batch 64, stop when the training loss improves by less than 1% over 10 epochs). Folds hold about 42.5k passages each.

- 1,925 passages have no canonical prompt id. They are grouped by the dataset's `prompt_group` instead, so they are spread over folds instead of all landing together.
- `train.py` now encodes each document sparsely once and densifies one batch at a time, so a 169k x 8000 dense matrix is never built. On the small-data folds it gives identical metrics to the old code.

### Results

Mean ± std over 5 folds.

| Model | Epochs | Train acc | Val acc | Val macro F1 | Val log loss |
|---|---|---|---|---|---|
| Log-linear | 24 | 0.387 ± 0.003 | 0.356 ± 0.004 | 0.281 ± 0.006 | 2.108 |
| MLP | about 105 | 0.867 ± 0.003 | **0.448 ± 0.002** | **0.414 ± 0.001** | **1.568** |
| SVM | 16 | 0.379 ± 0.004 | 0.349 ± 0.002 | 0.275 ± 0.003 | not defined |

ROPE with r = 0.01 (one accuracy point), same method as section 2:

| A | B | Mean A-B (acc) | P(A>B) | P(ROPE) | P(B>A) |
|---|---|---|---|---|---|
| Log-linear | MLP | -0.092 | 0.000 | 0.000 | 1.000 |
| Log-linear | SVM | +0.008 | 0.259 | 0.737 | 0.004 |
| MLP | SVM | +0.099 | 1.000 | 0.000 | 0.000 |

On macro F1 the picture is the same (MLP better than both with probability 1.00; log-linear and SVM equivalent with probability 0.76). Raw output: `big-data-approach/results/rope_validation_accuracy.json` and `rope_validation_macro_f1.json`.

### Comparison with the best low-data models

| Model | Training passages | Val acc | Val macro F1 |
|---|---|---|---|
| Low data, log-linear + token-indicator (best accuracy) | about 3,850 per fold | 0.304 ± 0.027 | 0.282 ± 0.025 |
| Low data, MLP + bag-of-words (best macro F1) | about 3,850 per fold | 0.297 ± 0.021 | 0.285 ± 0.021 |
| Big data, log-linear + token-indicator | about 170,000 per fold | 0.356 ± 0.004 | 0.281 ± 0.006 |
| Big data, MLP + token-indicator | about 170,000 per fold | 0.448 ± 0.002 | 0.414 ± 0.001 |

- **The MLP profits most from more data.** It gains about 14 points of accuracy and 13 points of macro F1 over the best low-data models. Chance is 5.6%.
- **Linear models gain much less, and the log-linear model's number is not a fair measure of what it can do.** Its train accuracy is only 0.387, so it is underfitting. With 44 times more batches per epoch, the 1% rule stopped training after 24 epochs while the loss was still falling slowly. The SVM stopped after 16 epochs, with the same issue. I kept the rule identical on purpose ("the exact same test"), but a looser rule (smaller tolerance or larger patience) would probably raise both. This has not been run.
- **The picture changes with data size.** With little data, the three models were close (section 2). With a lot of data, the MLP's extra capacity pays off and it separates from the linear models with probability 1.00. The linear models stay equivalent to each other (P(ROPE) about 0.74-0.76).
- **The MLP still overfits.** Train accuracy 0.867 against validation 0.448, a gap of 0.42. This is smaller than the 0.6-0.7 gap in the low-data runs, and the validation score is clearly higher.
- **The comparison is not controlled.** The datasets differ (different sources, generation settings and prompts), so the gain cannot be credited to data size alone. The folds are of the same kind (5-fold, grouped by prompt), so the standard deviations are comparable. The large gap to the low-data runs is far outside them.

### The SVM in both regimes

Same model, encoder (token-indicator), hyperparameters and stopping rule in both rows.

| Regime | Training passages per fold | Epochs | Train acc | Val acc | Val macro F1 |
|---|---|---|---|---|---|
| Small data | about 3,850 | 120-139 | 0.990 ± 0.002 | 0.292 ± 0.022 | 0.279 ± 0.019 |
| Big data | about 170,000 | 16 | 0.379 ± 0.004 | 0.349 ± 0.002 | 0.275 ± 0.003 |

On small data the SVM memorises (0.99 train against 0.29 val). On big data it is underfit (0.38 against 0.35) and stopped after 16 epochs. Validation accuracy rises by 5.7 points while macro F1 is flat, so the gain comes from classes it already handled well. The datasets differ, so this is not a controlled comparison.

### Regularization and capacity on the big data

Not run. The weight-decay and hidden-size experiments (section 3) were done on the small data only. On the big data the linear models are already underfit (train accuracy about 0.38), so stronger regularization could only hurt them. The big-data MLP (train 0.867, val 0.448) is the case where it could matter, and it is untested.

### Extra: fine-tuned transformer (outside the assignment's models)

`distilroberta-base` fine-tuned for 18-way classification (AdamW, learning rate 5e-5, batch 32, maximum 256 tokens, 3 epochs, 3.6 hours on an M5). This is not one of the assignment's models and it uses a single 80/10/10 split by prompt hash (169,184 train, 20,900 val, 22,539 test), not the cross-validation above, so it is only a reference point. Validation accuracy 0.519 (by epoch: 0.466, 0.504, 0.519), validation macro F1 0.479, test accuracy 0.534, test macro F1 0.488. In that split the 1,925 passages without a prompt id all fall into the test split.

Against the 5-fold MLP (val 0.448), the transformer is about 7 points higher, on a different split, with a training cost of 3.6 hours against about 75 minutes for the MLP's five folds. Validation accuracy was still rising after epoch 3.

## Interpretation

Neither L2 regularization (3a) nor a smaller MLP (3b) closes the gap: either the score stays the same with the training set still memorised, or it falls together with the training accuracy. So it is probably not plain overfitting by capacity. A likely explanation, which we have not tested, is that validation prompts are new topics. Topic words learned from training prompts then do not transfer. This would also fit tf-idf scoring slightly worse, since it up-weights rare, topic-specific tokens.

## Caveats

- 5 folds and one seed, so the uncertainty is large.
- The ROPE width is a choice that has to be justified. 0.01 gives a result worth discussing, while 0.05 makes everything equivalent.
- The weight-decay settings were chosen by looking at CV results. They are not a clean held-out estimate.
- The weight-decay and hidden-size runs are not included in the ROPE comparison.
- `data/folds.csv` and the per-fold tokenizers had to be regenerated (`scripts/split_data.py`, `scripts/preprocess_data.py`) before training, because the existing `data/` was stale.

## Not done

- A deeper MLP or an MLP with dropout as the extra architecture (only the hidden width was varied, section 3b). The SVM is our extra architecture for now.
- More data in the main experiment: the split excludes the competition prompts, the pilot and the Kimi-only rerun on purpose. Section 4 instead uses a separate, larger dataset.
- MLP dropout, L1 regularization, a smaller vocabulary and other regularizers. Regularization and hidden-size sweeps on the big data. A looser stopping rule for the big-data linear models, which are underfit.
- The notebook also asks for a record of AI assistance, the sources used, and backup slides for the oral. These are not part of this file.

## Part 2 embedding comparison

The comparison used all 4,818 passages from 282 prompts and the same five folds in
`data/folds.csv` as the BPE baseline. The split keeps all responses to each prompt in the
same fold and balances the 18 model labels. Each run trains on four folds and validates
on the fifth, so each passage is held out once.

These runs used the 300-dimensional Gensim word-vector exports, including the fixed
fastText table. They used NLTK sentence and word tokenization followed by lowercasing,
frozen vectors, normalized mean pooling, and a log-linear classifier. The training defaults
match Part 1, including seed 42 and the training-loss stopping rule. The original report is
in `artifacts/embeddings/comparison.json`. See the fastText binary comparison below for the
current implementation and its results.

| Tokenization | GloVe token coverage | GloVe distinct-token coverage | fastText token coverage | fastText distinct-token coverage |
|---|---:|---:|---:|---:|
| Whitespace | 78.36% | 37.78% | 86.62% | 49.41% |
| Lowercase + whitespace | 83.67% | 41.60% | 86.42% | 47.36% |
| NLTK word tokenization, then lowercase | 96.45% | 73.91% | 98.08% | 82.87% |
| NLTK sentence and word tokenization, then lowercase | 99.53% | 92.58% | 99.23% | 94.25% |

Both NLTK rows lowercase after tokenization. Lowercasing first in the word-only run
produced identical lookup tokens on all 4,818 passages, so its original scores still apply.
Enabling sentence splitting removed all 319 occurrences of `it.` as a single token. NLTK
can use the original case to find sentence boundaries. The configuration keeps punctuation
and stopwords as possible style cues. For fastText, NLTK's quote markers and the contraction
token `n't` still miss the vocabulary. Both Gensim downloads use fixed word tables.
Missing tokens contribute zero vectors.

Both models placed `however` near `although` and `though`, and `therefore` near `thus` and
`consequently`. GloVe placed the semicolon near a colon, comma, and period. fastText placed
the em dash near the en dash and hyphen. Some neighbors were weaker style matches, such as
GloVe's `overall` near `improved` and fastText's `however` near `therefore`.

| Embedding | Mean accuracy before sentence splitting | Mean accuracy after (± fold SD) | Mean macro F1 after (± fold SD) |
|---|---:|---:|---:|
| `glove-wiki-gigaword-300` | 19.02% | 18.56% ± 1.61 pp | 14.49% ± 1.23 pp |
| `fasttext-wiki-news-subwords-300` | 19.54% | 19.60% ± 1.82 pp | 14.99% ± 1.32 pp |

With sentence splitting, coverage increased for both models. Mean accuracy fell by 0.45
percentage points for GloVe and rose by 0.06 points for fastText. In the new runs, fastText
led by 1.03 points on average. The paired differences were +2.98, +1.54, +1.47, −0.10,
and −0.73 points. With the same one-point ROPE as Part 1, the Bayesian correlated t-test
assigns 51.2% probability to a fastText advantage of more than one point and 43.4% to
practical equivalence. Both runs use one training seed. The Part 1 log-linear BPE
token-indicator baseline scored 30.4% mean accuracy on these folds. The embedding releases
also differ in pretraining corpus and vocabulary, so the comparison measures the released
embeddings as a whole.

## Part 2 mean pooling vs LSTM

Both encoders used frozen GloVe 300d, **NLTK sentence splitting → NLTK word tokenization
→ lowercase**, the same five prompt-grouped folds, seed 42, and the Part 1 optimizer and
training-loss stopping rule. Both used a linear classifier. Mean pooling averaged the vectors.
The one-layer LSTM used 64 hidden units and the final hidden state. Both normalized the
passage representation before classification.

| Encoder | Mean train accuracy | Mean validation accuracy (± fold SD) | Mean validation macro F1 |
|---|---:|---:|---:|
| Mean pooling | 21.38% | 18.56% ± 1.61 pp | 14.49% |
| LSTM | 82.25% | 16.02% ± 3.39 pp | 14.41% |

The LSTM won one fold and was 2.55 percentage points lower on average. With a one-point
ROPE, the posterior probability that mean pooling led by more than one point was 69.1%.
Four LSTM folds reached at least 99.7% training accuracy while validation accuracy stayed
between 16.6% and 19.0%. In fold 0, validation loss reached 2.25 at epoch 16 and rose to
4.19 by epoch 112, while training loss fell from 1.97 to 0.03. Fold 3 stopped after 18
epochs with 11.5% train accuracy, which shows the training-loss rule also stopped an
underfit run. The run files and curves are in `artifacts/embeddings/glove-lstm/`.

Reproduce with `uv run ntl-train --encoder glove --pooling lstm --output artifacts/embeddings/glove-lstm`.

## Part 2 design experiments

The sweep uses the same tokenizer and five folds. Every setting uses 20 Adam epochs,
batch size 128, weight decay `1e-4`, and seed 42. This fixed budget applies to both encoders.
The word vocabulary is built from each training split for all settings, so validation-only
words use the same unknown token when comparing frozen, fine-tuned, and randomly initialized
embeddings. The 25% and 50% subsets use nested groups of training prompts and keep the full
validation fold. At a fixed epoch count, smaller subsets also get fewer optimizer updates.

GloVe 50d and 300d are native releases trained on Wikipedia + Gigaword. They test width
within GloVe. FastText is the native 300d Wiki News release and provides a same-width
model comparison with GloVe 300d. The remaining changes use GloVe 50d as their reference.

| Setting | Mean pooling accuracy | LSTM accuracy |
|---|---:|---:|
| GloVe 50d, frozen | 11.04% | 12.72% |
| GloVe 300d, frozen | 15.59% | 12.54% |
| fastText 300d, frozen | 16.25% | 9.65% |
| GloVe 50d, MLP head for pooling or 32 hidden units for LSTM | 13.44% | 13.34% |
| GloVe 50d, Adam learning rate `3e-3` instead of `1e-3` | 12.85% | 13.86% |
| GloVe 50d, fine-tuned | 15.01% | 14.75% |
| 50d, randomly initialized and trained | 25.24% | 16.73% |
| GloVe 50d, 50% of training prompts | — | 10.97% |
| GloVe 50d, 25% of training prompts | — | 8.68% |

- **Embedding size.** GloVe 300d improved mean pooling by 4.55 percentage points and won
  all five folds. It changed LSTM accuracy by −0.18 points on average, with differences
  in both directions. This comparison uses two published checkpoints from one family.
- **Architecture and optimizer.** With 50d frozen GloVe, the pooling MLP head gained 2.40
  points over a linear head. The 32-unit LSTM gained 0.62 points over 64 units on average,
  with mixed fold results. Raising Adam's learning rate to `3e-3` gained 1.81 points for
  pooling and 1.14 for the LSTM at the fixed epoch budget.
- **Pretraining.** Random initialization beat fine-tuned GloVe by 10.23 points for pooling
  and won all five folds. For the LSTM, random initialization averaged 1.98 points higher,
  but fine-tuned GloVe won three folds. The best pooling setting scored 25.24%, below the
  earlier 30.4% BPE token-indicator baseline. These runs give no clear benefit from
  pretrained initialization.
- **Data size.** With frozen 50d GloVe and 64 hidden units, using 25%, 50%, and 100% of the
  training prompts gave 8.68%, 10.97%, and 12.72% accuracy. The full set beat each smaller
  set on all five validation folds.
- **LSTM vs. pooling.** The 50d frozen LSTM beat its matched pooling reference by 1.68 points
  and won all five folds at 20 epochs. At 300d, mean pooling won all five matched folds
  for both embeddings. The best LSTM scored 16.73% accuracy and 13.80% macro F1. The best
  pooling model scored 25.24% accuracy and 19.93% macro F1. The architectural advantage
  depended on the embedding setup, and recurrence did not improve on the best pooling model.

The sweep uses one training seed and a fixed budget. Its learning-rate and architecture
effects describe performance at 20 epochs. The longer GloVe comparison above shows why
training-loss convergence alone gave poor validation performance for the LSTM.

Run `uv run python scripts/embedding_sweep.py --encoder mean` and
`uv run python scripts/embedding_sweep.py --encoder lstm --device mps` to reproduce.
Use `--device cpu` on machines without Apple GPU support. Per-fold scores and loss histories
are in `artifacts/embeddings/sweep/`, with aggregate scores in `summary.json`.

## Part 2 tuned mean-pooling classifiers

The three logistic-regression baselines use frozen GloVe 50d, GloVe 300d, or fastText 300d.
They share the same five prompt-grouped folds, sentence splitting followed by NLTK word
tokenization and lowercasing, normalized mean pooling, and all training settings. Adam uses
learning rate `0.001`, weight decay `1e-4`, and batch size 128. The word vocabulary is fitted
on each training split, as in the design sweep above.

For frozen GloVe 300d and fastText 300d, the same MLP grid combines hidden layers of
`128` or `256 → 128` units, Adam learning rates `0.001`, `0.003`, or `0.01`, and dropout
of `0` or `0.25`. The initial grid trains for 100 epochs and compares checkpoints at epochs
20, 50, and 100. Both embeddings selected one 128-unit layer, learning rate `0.003`, and no
dropout, with the best scores at epoch 100. The follow-up trains both winners and all three
logistic baselines for 200 epochs with seeds 42, 17, and 73. It compares checkpoints at
epochs 100, 150, and 200 for each winner. The final epoch for each is selected by mean
validation accuracy with seed 42, and the other seeds use that same epoch. The selection
and repeats use the same folds, so the scores describe performance on these folds and seed
sensitivity. An independent test would be needed for a fresh estimate after selection.

At the shared `0.001` learning rate and 100 epochs, the 128-unit MLP with no dropout improved
seed-42 accuracy from 19.41% to 23.88% for GloVe 300d and from 20.44% to 27.29% for fastText
300d. This comparison isolates the classifier change. At epoch 100, averaging matched
settings across the grid, `0.003` beat `0.001` by 1.36 points for GloVe and 2.24 for fastText.
The two-layer head scored 2.39 and 3.30 points lower on average than the one-layer head.
Dropout `0.25` also reduced average accuracy in this grid.

Both winners reached their highest seed-42 score at epoch 200 in the follow-up. All rows
below use 200 epochs and average the same five folds over three seeds.

| Frozen embeddings | Classifier | Adam learning rate | Mean accuracy | Mean macro F1 |
|---|---|---:|---:|---:|
| GloVe 50d | Logistic regression | 0.001 | 14.32% | 10.62% |
| GloVe 300d | Logistic regression | 0.001 | 20.98% | 17.31% |
| fastText 300d | Logistic regression | 0.001 | 21.68% | 17.20% |
| GloVe 300d | 128-unit MLP | 0.003 | 26.08% | 23.95% |
| fastText 300d | 128-unit MLP | 0.003 | **30.17%** | **27.12%** |

Moving from 50d to 300d GloVe gained 6.66 accuracy points with the same logistic settings.
The tuned MLP gained 5.09 points over its GloVe 300d logistic baseline and 8.50 over its
fastText 300d baseline. These gains combine the classifier and learning-rate changes. At the
same MLP settings, fastText led GloVe by 4.10 points and led on all five seed-averaged folds.

The best frozen embedding pipeline came close to the Part 1 BPE token-indicator baseline
(30.4% accuracy and 28.2% macro F1 under its original training rule). The three fastText MLP
seed means ranged from 29.90% to 30.51%. Its mean of 30.17% gives no clear improvement over
Part 1. This compares complete text representations. The frozen comparison cannot isolate
pretrained initialization from the representation itself.

Run `uv run python scripts/embedding_sweep.py --pooling-grid --epochs 100` to apply the
same grid with the current embeddings. For the 200-epoch follow-up, repeat this command
with seeds `42`, `17`, and `73`:

```sh
uv run python scripts/embedding_sweep.py --pooling-grid --epochs 200 --seed 42 \
  --setting glove50 --setting glove300 --setting fasttext300 \
  --setting glove300-mlp128-lr0.003-drop0 --setting fasttext300-mlp128-lr0.003-drop0
```

Fold scores and learning histories are in `artifacts/embeddings/sweep/`. The follow-up
summary is in `artifacts/embeddings/sweep/pooling-refine/pooling-mlp/summary.json`.

## Part 2 tuned LSTM on 300d embeddings

The recurrent grid uses frozen GloVe 300d and fastText 300d with the same five folds,
training-fold vocabulary, and sentence splitting → NLTK word tokenization → lowercase as
the tuned mean-pooling comparison. Each model has a one-layer 64-unit LSTM and a 128-unit
MLP classifier. The passage representation uses one of three readouts:

- **Last state:** take the last non-padding LSTM output.
- **Mean states:** average the LSTM outputs over non-padding positions.
- **Hybrid:** concatenate the mean LSTM states with the mean word vectors. Normalize each
  before concatenation so both parts contribute to the classifier input.

The initial grid uses seed 42 and 20 Adam epochs, with batch size 128 and weight decay `1e-4`.
It tests learning rates `0.001` and `0.003` for last state and mean states. The hybrid uses
`0.003`, so it can be compared against the other readouts at that rate. Candidate checkpoints
are epochs 5, 10, 15, and 20. Each setting is compared at the same checkpoint across folds.

| Readout | Adam learning rate | GloVe 300d accuracy | fastText 300d accuracy |
|---|---:|---:|---:|
| Last state | 0.001 | 15.33% | 11.53% |
| Last state | 0.003 | 17.41% | 8.43% |
| Mean states | 0.001 | 24.07% | 25.89% |
| Mean states | 0.003 | **26.40%** | **26.19%** |
| Hybrid | 0.003 | 24.12% | 25.37% |

At epoch 20, averaging LSTM states beat the last-state readout on every fold for both
embeddings and learning rates. Raising the learning rate from `0.001` to `0.003` helped the
mean readout by 2.33 points for GloVe and 0.30 for fastText. The higher rate hurt fastText
with the last-state readout. Adding the direct word-vector mean to the recurrent mean gave
lower accuracy for both embeddings at the matched rate. Across the four checkpoint choices,
the highest initial mean accuracy was 26.43% for GloVe mean states at epoch 15.

At the same 20 epochs, seed 42, learning rate `0.003`, and 128-unit classifier, ordinary
mean pooling scored 20.21% with GloVe and 21.39% with fastText. The mean-state LSTMs were
ahead by 6.19 and 4.80 points at that checkpoint. Each LSTM epoch uses more computation,
and the tuned mean-pooling comparison also ran through epoch 200.

With GloVe mean states at `0.003`, mean training loss fell from 2.02 at epoch 5 to 0.93 at
epoch 20. Mean validation loss rose from 2.15 to 2.83. Accuracy barely changed from epochs
15 to 20. The [learning curves](artifacts/embeddings/sweep/lstm-grid/learning-curves.png)
show this overfitting directly.

The follow-up uses the leading GloVe setting with frozen, fine-tuned, or randomly initialized
embeddings, and also tests 25% and 50% of the training prompts with full validation folds.
Two regularization settings test 25% dropout in the classifier or 15% chance of masking each
word vector during training. All follow-up settings run for 30 epochs. The leading frozen
fastText setting also runs for 30 epochs.

At epoch 25, 15% word dropout raised GloVe validation accuracy from 25.47% to 27.65%.
It lowered validation loss from 3.45 to 2.43 at the same checkpoint. The 25% classifier
dropout setting scored 25.85% and reduced validation loss to 3.05. With word dropout,
validation loss still rose later in training, so it reduced the overfitting seen in the curves.
The same word dropout hurt fastText: at epoch 25, accuracy was 17.83% versus 29.00% with
dropout off. It was useful for GloVe in this experiment, with a different effect on fastText.

| GloVe 300d mean-state LSTM setting | Epoch-25 accuracy |
|---|---:|
| Frozen vectors, full training fold | 25.47% |
| Fine-tuned vectors, full training fold | 21.46% |
| Randomly initialized and trained vectors, full training fold | 24.01% |
| Frozen vectors, 50% of training prompts | 21.48% |
| Frozen vectors, 25% of training prompts | 19.82% |

At this checkpoint, frozen GloVe led both trainable variants. Fine-tuning also had higher
validation loss (6.07 versus 3.45). With frozen vectors, increasing the number of training
prompts raised mean accuracy. The 25% subset used 972 passages per fold, the 50% subset
used 1,926–1,944, and the full training folds used 3,846–3,864. Validation stayed fixed.
At the same epoch count, a smaller subset also gets fewer Adam updates.

The top seed-42 result was 29.00% for frozen fastText 300d with mean LSTM states, Adam
at `0.003`, and both dropout options off. Epoch 25 was the best common checkpoint among
epochs 5, 10, 15, 20, 25, and 30. Repeating that configuration and checkpoint with seeds 17
and 73 gave 26.27% and 27.23%. The table averages all three seeds on the same five folds.

| Frozen fastText 300d and 128-unit MLP | Epochs | Mean accuracy | Mean macro F1 |
|---|---:|---:|---:|
| Mean word-vector pooling | 25 | 23.23% | 17.95% |
| Mean LSTM states | 25 | 27.50% | 23.49% |
| Best tuned mean word-vector pooling | 200 | 30.17% | 27.12% |

At 25 epochs, the LSTM led by 4.27 points and won all five seed-averaged folds. This
compares equal epoch counts, with more computation per epoch for the LSTM. Against the
best tuned pooling run, the LSTM was 2.67 points lower and won one of five seed-averaged
folds. Using the one-point ROPE, the posterior probability that the best pooling run led
by more than one point was 86.9%. These are comparisons on the folds used for tuning.

To run the recurrent experiments with the current embeddings on Apple GPU, use:

```sh
uv run python scripts/embedding_sweep.py --encoder lstm --device mps --lstm-grid --epochs 20
uv run python scripts/embedding_sweep.py --encoder lstm --device mps \
  --lstm-reference glove300-mean-lr0.003 --epochs 30
uv run python scripts/embedding_sweep.py --encoder lstm --device mps \
  --lstm-reference fasttext300-mean-lr0.003 --epochs 30 \
  --setting fasttext300-mean-lr0.003 --setting word-dropout
```

To repeat the finalist, use the last command with `--epochs 25`, keep only
`--setting fasttext300-mean-lr0.003`, and run once with `--seed 17` and once with `--seed 73`.
Use `--device cpu` without Apple GPU. Raw folds and curves are in
`artifacts/embeddings/sweep/lstm-grid/` and `artifacts/embeddings/sweep/lstm-followup/`.
The three-seed comparison is saved as `artifacts/embeddings/sweep/lstm-followup/summary.json`.

## Part 2 original fastText binary

The fastText encoder now loads `wiki-news-300d-1M-subword.bin`. The binary has two
million explicit words and can compose vectors from character four-grams for other
tokens. The Gensim word-vector export has about one million explicit words. The comparison
uses the same 4,818 passages, five prompt-grouped folds, and NLTK sentence splitting and
word tokenization followed by lowercasing. The vectors stay frozen in all runs. The tuned
classifiers also use the same training-fold vocabulary rule and seeds 42, 17, and 73.

| fastText 300d lookup | Token coverage | Distinct-token coverage |
|---|---:|---:|
| Gensim export, direct lookup | 99.23% | 94.25% |
| Original binary, direct lookup | 99.39% | 95.43% |
| Original binary, direct lookup plus subwords | **100.00%** | **100.00%** |

The larger vocabulary directly covers 1,010 more occurrences. Subwords cover the remaining
3,926 occurrences across 1,062 distinct tokens, including NLTK's opening and closing quote
markers and `n't`. The binary still places `therefore` near `thus` and the em dash near
the en dash. The closest neighbor of `however` becomes the spelling variant `howevever`,
followed by `though` and `although`. The 21,907 tokens shared by both releases have cosine
similarity of 1 to rounding precision for their paired vectors. Their vector lengths differ
by word, with a median binary-to-export ratio of 0.40. The swap therefore changes word
weights in the document representation as well as coverage.

| Classifier and representation | Epochs | Export accuracy | Binary accuracy | Export macro F1 | Binary macro F1 |
|---|---:|---:|---:|---:|---:|
| Mean pooling + logistic regression, original stopping rule | Varies | 19.60% | 19.12% | 14.99% | 14.30% |
| Mean pooling + logistic regression | 200 | 21.68% | 21.23% | 17.20% | 16.63% |
| Mean pooling + 128-unit MLP | 25 | 23.23% | **25.45%** | 17.95% | **20.82%** |
| Mean pooling + 128-unit MLP | 200 | 30.17% | **30.21%** | 27.12% | **27.25%** |
| Mean states of a 64-unit LSTM + 128-unit MLP | 25 | 27.50% | 24.02% | 23.49% | 20.34% |

The first row uses seed 42. All other rows average three seeds over the five folds. The
tuned 200-epoch pooling score was effectively unchanged and stayed close to the Part 1
BPE token-indicator baseline of 30.4%. At 25 epochs, switching to the binary improved
mean pooling on all five seed-averaged folds. The LSTM lost 3.48 accuracy points overall.
Its seed means ranged from 22.21% to 27.09%, and two of the 15 runs stalled near chance.
Repeating those two runs reproduced the same validation accuracies, 6.25% and 5.86%.
At 25 epochs, binary mean pooling was 1.44 points ahead of the binary LSTM. Both use
the previously selected settings.

Run the binary comparison with these commands. Repeat both sweep commands with seeds
`17` and `73`. Use `--device cpu` when Apple GPU support is unavailable.

```sh
uv run python scripts/compare_embeddings.py --embedding fasttext \
  --output artifacts/embeddings/fasttext-binary/linear-comparison.json
uv run python scripts/embedding_sweep.py --pooling-grid --epochs 200 --seed 42 \
  --setting fasttext300 --setting fasttext300-mlp128-lr0.003-drop0
uv run python scripts/embedding_sweep.py --encoder lstm --device mps --epochs 25 --seed 42 \
  --lstm-reference fasttext300-mean-lr0.003 --setting fasttext300-mean-lr0.003
```

### FastText mean pooling versus LSTM with early stopping

Both models use frozen fastText 300d vectors, a 128-unit MLP, and Adam at `0.003`.
Each run has a 200-epoch cap and uses the Part 1 stopping rule. Training stops after
10 consecutive epochs without a 1% relative improvement over the best tracked training
loss. We score the final epoch in each run and average five prompt-grouped folds over
seeds 42, 17, and 73.

| Encoder | Epochs run (mean) | Mean accuracy | Mean macro F1 |
|---|---:|---:|---:|
| Mean word-vector pooling | 81–132 (101.9) | **29.11%** | **25.20%** |
| Mean states of a 64-unit LSTM | 12–200 (155.6) | 24.27% | 22.48% |

Mean pooling led by 4.84 accuracy points and scored higher on four of five seed-averaged
folds. Two LSTM runs stopped near chance at epoch 12, and two reached the 200-epoch cap.
For the 13 LSTM runs that passed epoch 25, mean training loss fell from 1.78 at epoch 25
to 0.27 at the final epoch, while mean validation loss rose from 2.03 to 4.53. The
training-loss stopping rule let training continue as validation loss rose.

To repeat the runs, use the sweep commands above with `--early-stopping` and `--epochs 200`.
Keep `--setting fasttext300-mlp128-lr0.003-drop0` for the mean-pooling command. The fold
histories and aggregate results are in `artifacts/embeddings/fasttext-binary/early-stopping/`.

### GloVe and fastText LSTMs at 200 fixed epochs

Both models use frozen 300d vectors, a one-layer 64-unit LSTM, and mean pooling over
the LSTM states. A 128-unit MLP classifies the pooled representation. Adam uses learning
rate `0.003`. Every fold runs for 200 epochs. We score epoch 200 on the same five
prompt-grouped folds and average over seeds 42, 17, and 73.

| Frozen embedding | Epochs | Mean accuracy | Mean macro F1 |
|---|---:|---:|---:|
| fastText 300d binary | 200 | **25.15%** | **23.85%** |
| GloVe 300d | 200 | 23.44% | 22.71% |

FastText led by 1.71 accuracy points and scored higher on four of five seed-averaged
folds. The learning curves showed overfitting in both models. From epoch 25 to 200,
mean validation loss rose from 2.14 to 4.81 for fastText and from 3.44 to 5.87 for GloVe,
while the training loss fell for both. Fold histories are in
`artifacts/embeddings/fasttext-binary/fixed-200/lstm-grid/`.

Repeat the command with `--seed 17` and `--seed 73` for the other seeds.

```sh
uv run python scripts/embedding_sweep.py --encoder lstm --device mps --epochs 200 \
  --output artifacts/embeddings/fasttext-binary/fixed-200 --lstm-grid --seed 42 \
  --setting glove300-mean-lr0.003 --setting fasttext300-mean-lr0.003
```

### Fine-tuning word embeddings for 200 epochs

Each model starts with its pretrained 300d word vectors and updates those vectors during
training. The fastText binary supplies the initial word vectors, including vectors formed
from subwords. Fine-tuning updates each word's vector. The experiment uses the same LSTM,
classifier, optimizer, folds, seeds, and fixed 200-epoch budget as the frozen comparison.

| Embedding initialization | Frozen accuracy | Trainable accuracy | Frozen macro F1 | Trainable macro F1 |
|---|---:|---:|---:|---:|
| fastText 300d binary | 25.15% | 21.46% | 23.85% | 20.81% |
| GloVe 300d | 23.44% | 20.72% | 22.71% | 20.14% |

Fine-tuning reduced accuracy by 3.69 points for fastText and 2.72 for GloVe. Accuracy fell
on four of five seed-averaged folds for fastText and all five for GloVe. Mean validation
loss also rose for both. The fold histories are under
`artifacts/embeddings/fasttext-binary/fixed-200/lstm-followup/`.

Repeat this command for `--seed 17` and `--seed 73` and for
`--lstm-reference fasttext300-mean-lr0.003`.

```sh
uv run python scripts/embedding_sweep.py --encoder lstm --device mps --epochs 200 \
  --output artifacts/embeddings/fasttext-binary/fixed-200 --seed 42 \
  --lstm-reference glove300-mean-lr0.003 --setting finetuned
```

### Word dropout in the frozen fastText LSTM

The experiment compares word dropout of 0% and 25% with frozen fastText 300d binary
vectors. Both settings use a 64-unit LSTM, mean-state readout, a 128-unit MLP, and Adam
at `0.003`. Both run for 200 fixed epochs on the same five prompt-grouped folds with
seed 42. During training, word dropout independently zeroes entire input vectors with
the given probability. Validation uses the full input.

| Word dropout | Training loss, epoch 25 → 200 | Validation loss, epoch 25 → 200 | Epoch-200 accuracy | Epoch-200 macro F1 |
|---|---:|---:|---:|---:|
| 0% | 1.99 → 0.12 | 2.19 → 5.01 | 26.45% | 25.51% |
| 25% | 2.04 → 1.29 | 2.13 → 2.30 | 27.56% | 25.67% |

Word dropout lowered epoch-200 validation loss on all five folds and raised accuracy on
three. The mean accuracy gain was 1.12 percentage points. Mean validation loss reached
its lowest point at epoch 46 without word dropout and at epoch 76 with it. Loss still
rose later in the dropout run. The learning curves show a smaller rise in validation loss
as training loss falls. This experiment uses one seed. Fold histories, the summary, and
the plot are in `artifacts/embeddings/fasttext-binary/word-dropout-25/`.

```sh
uv run python scripts/embedding_sweep.py --encoder lstm --device mps --epochs 200 \
  --output artifacts/embeddings/fasttext-binary/word-dropout-25 --seed 42 \
  --lstm-reference fasttext300-mean-lr0.003 --word-dropout-probability 0.25 \
  --setting fasttext300-mean-lr0.003 --setting word-dropout
```

## Part 2 error analysis

The question: which properties of a passage make it easier or harder to attribute, and do
these properties affect the three main designs differently?

### Systems and data

| System | Representation | Classifier | Accuracy (seeds 42 / 17 / 73) |
|---|---|---|---:|
| `bpe-indicator` | BPE token indicators, 8,000 tokens (Part 1) | Log-linear | 30.45% (one Part 1 run) |
| `glove-pool-mlp` | Frozen GloVe 300d, mean pooling | 128-unit MLP, Adam `0.003`, 200 epochs | 25.97 / 26.75 / 25.51% |
| `glove-lstm` | Frozen GloVe 300d, mean of 64-unit LSTM states, 15% word dropout | 128-unit MLP, Adam `0.003`, 25 epochs | 27.94 / 26.36 / 27.48% |

These are the leading settings of the earlier GloVe sweeps. We used GloVe and not fastText
because the fastText binary is a 4.5 GB download. The GloVe accuracies are close to the
earlier sweep results (26.08% and 27.65%). The three systems use their own tuned training
settings, so the comparison is between tuned systems, not between architectures with all
other settings equal.

Each passage gets one prediction from each system: the model of fold k predicts the
passages of fold k, which it did not train on. So all 4,818 passages are analysis
data. There is no separate devtest set, because the project uses all passages in
cross-validation (see "Cross-validation folds" in the README). The settings were chosen
on these same folds, so the accuracies are slightly optimistic. The feature weights
below compare systems with each other, which is less affected by this.

### Features

| Feature | Definition | Reason to test it |
|---|---|---|
| `log_words` | log of the word count (whitespace-separated words) | A longer passage gives more evidence. The log prevents the longest passages from dominating. |
| `sentence_length` | mean words per sentence (NLTK sentence splitter) | Sentence structure is a style property. |
| `em_dash_rate` | em dashes (`—`) per 100 words | A known LLM style marker. |
| `semicolon_colon_rate` | semicolons and colons per 100 words | Punctuation style. |

The four features are only weakly correlated. The largest correlation is 0.22, between
`log_words` and `sentence_length`. So each weight measures mostly its own feature.

The punctuation features differ strongly between model families:

| Family | Em dashes per 100 words | Semicolons and colons per 100 words | BPE accuracy |
|---|---:|---:|---:|
| Claude | 1.27 | 0.83 | 33% |
| GLM | 1.54 | 1.30 | 12% |
| Kimi | 1.37 | 1.28 | 6% |
| GPT | 0.42 | 1.24 | 32% |
| Qwen | 0.14 | 1.02 | 26% |
| Gemini | 0.25 | 0.47 | 40% |

GLM and Kimi, the two hardest families, have the most em dashes and the most semicolons
and colons. Gemini, the easiest family, has few of both.

### Method

We fit the Bayesian logistic regression of the tutorial (`part2/analysis.py`), with NUTS in
NumPyro (4 chains, 1,000 warm-up and 1,000 kept samples each). For passage n and system s:

`logit P(correct) = phi + delta_s + w_s · f_n`, with Laplace(0, 1) priors on all parameters.

The outcome is 1 when the system names the correct model, and 0 when it does not. The
features are standardized, so a weight is the change in the log-odds of a correct
prediction for one standard deviation of the feature. Each system has its own weights. All
fits had 0 divergences and r_hat ≤ 1.011.

**ROPE.** We use a region of practical equivalence of ±0.05 for weights and for differences
of weights (`--rope 0.05`). At an accuracy p, a small weight w changes accuracy by
approximately p(1 − p) · w. At p ≈ 0.28 this is 0.2 · w, so ±0.05 is approximately ±1 accuracy
point per standard deviation of the feature. This is the same ROPE as the Part 1 accuracy
comparison. For each weight we report the posterior mass below, inside and above the ROPE.
We call an effect practical when more than 0.95 of the mass is on one side of the ROPE, and
probable when more than 0.65 is on one side, in all three seeds. The ROPE translation is
approximate, because the accuracy of each system and each passage is different.

**The author model is a confounder.** Some labels are much easier than others (per-label
accuracy of `bpe-indicator` is from 6% for `claude-fable-5` and `kimi-k3` to 57% for
`gemini-3.1-pro-preview`), and the punctuation features differ by family (table above).
Without a control, a feature can look important only because it marks an easy or a hard
model. For this reason the main analysis (`--control-label`) adds 17 indicators of the true
author model. The feature weights then compare passages of the same model.

Weights for seed 42, without and with the label control (posterior mass below / inside /
above the ROPE in brackets):

| Feature | System | Without label control | With label control |
|---|---|---|---|
| `em_dash_rate` | BPE | −0.14 (1.00 / 0.00 / 0.00) | +0.04 (0.02 / 0.62 / 0.36) |
| | pool | −0.16 (1.00 / 0.00 / 0.00) | +0.01 (0.09 / 0.73 / 0.18) |
| | LSTM | −0.14 (1.00 / 0.00 / 0.00) | +0.04 (0.02 / 0.61 / 0.37) |
| `semicolon_colon_rate` | BPE | −0.10 (0.95 / 0.05 / 0.00) | −0.01 (0.18 / 0.79 / 0.04) |
| | pool | −0.03 (0.32 / 0.67 / 0.00) | −0.00 (0.11 / 0.81 / 0.09) |
| | LSTM | −0.08 (0.79 / 0.21 / 0.00) | −0.02 (0.19 / 0.78 / 0.03) |
| `sentence_length` | BPE | +0.06 (0.00 / 0.41 / 0.59) | +0.10 (0.00 / 0.09 / 0.91) |
| `log_words` | BPE | +0.25 (0.00 / 0.00 / 1.00) | +0.28 (0.00 / 0.00 / 1.00) |

Without the control, em dashes clearly seem to cause errors for all three systems, and
semicolons and colons for BPE. With the control, the mass below the ROPE is 0.10 or less
for em dashes and 0.22 or less for semicolons and colons, in all fits. The negative effects
came from the author model: these marks are frequent in GLM and Kimi, which are the hardest
families. A passage does not become harder because it has more em dashes or semicolons.

### Results with label control

Weights with 90% posterior intervals and the ROPE masses (below / inside / above). The BPE
rows are the same Part 1 run for every seed.

| Feature | System | Seed 42 | Seed 17 | Seed 73 |
|---|---|---|---|---|
| `log_words` | BPE | +0.28 [+0.23, +0.34]<br>0.00 / 0.00 / 1.00 | +0.28 [+0.23, +0.34]<br>0.00 / 0.00 / 1.00 | +0.28 [+0.22, +0.34]<br>0.00 / 0.00 / 1.00 |
| | pool | +0.19 [+0.13, +0.26]<br>0.00 / 0.00 / 1.00 | +0.19 [+0.13, +0.25]<br>0.00 / 0.00 / 1.00 | +0.19 [+0.13, +0.25]<br>0.00 / 0.00 / 1.00 |
| | LSTM | +0.19 [+0.14, +0.25]<br>0.00 / 0.00 / 1.00 | +0.27 [+0.21, +0.33]<br>0.00 / 0.00 / 1.00 | +0.20 [+0.14, +0.26]<br>0.00 / 0.00 / 1.00 |
| `sentence_length` | BPE | +0.10 [+0.04, +0.17]<br>0.00 / 0.09 / 0.91 | +0.10 [+0.04, +0.17]<br>0.00 / 0.09 / 0.91 | +0.10 [+0.04, +0.17]<br>0.00 / 0.10 / 0.90 |
| | pool | +0.22 [+0.15, +0.29]<br>0.00 / 0.00 / 1.00 | +0.17 [+0.11, +0.24]<br>0.00 / 0.00 / 1.00 | +0.20 [+0.13, +0.27]<br>0.00 / 0.00 / 1.00 |
| | LSTM | +0.19 [+0.12, +0.25]<br>0.00 / 0.00 / 1.00 | +0.16 [+0.10, +0.23]<br>0.00 / 0.00 / 1.00 | +0.20 [+0.14, +0.27]<br>0.00 / 0.00 / 1.00 |
| `em_dash_rate` | BPE | +0.04 [−0.03, +0.10]<br>0.02 / 0.62 / 0.36 | +0.04 [−0.03, +0.10]<br>0.02 / 0.62 / 0.36 | +0.03 [−0.03, +0.10]<br>0.02 / 0.63 / 0.35 |
| | pool | +0.01 [−0.06, +0.08]<br>0.09 / 0.73 / 0.18 | +0.09 [+0.02, +0.16]<br>0.00 / 0.20 / 0.80 | +0.06 [−0.01, +0.13]<br>0.00 / 0.44 / 0.55 |
| | LSTM | +0.04 [−0.03, +0.10]<br>0.02 / 0.61 / 0.37 | +0.08 [+0.01, +0.15]<br>0.00 / 0.22 / 0.78 | +0.03 [−0.04, +0.09]<br>0.03 / 0.70 / 0.27 |
| `semicolon_colon_rate` | BPE | −0.01 [−0.07, +0.04]<br>0.18 / 0.79 / 0.04 | −0.01 [−0.07, +0.04]<br>0.16 / 0.80 / 0.03 | −0.01 [−0.07, +0.04]<br>0.16 / 0.80 / 0.04 |
| | pool | −0.00 [−0.07, +0.06]<br>0.11 / 0.81 / 0.09 | −0.02 [−0.08, +0.04]<br>0.22 / 0.75 / 0.03 | +0.01 [−0.05, +0.07]<br>0.06 / 0.79 / 0.15 |
| | LSTM | −0.02 [−0.08, +0.04]<br>0.19 / 0.78 / 0.03 | −0.01 [−0.07, +0.06]<br>0.12 / 0.81 / 0.07 | +0.00 [−0.06, +0.06]<br>0.07 / 0.83 / 0.10 |

Posterior mass of differences of weights (first system minus second) with respect to the
ROPE (below / inside / above):

| Feature | Difference | Seed 42 | Seed 17 | Seed 73 |
|---|---|---|---|---|
| `log_words` | BPE − pool | 0.00 / 0.21 / 0.79 | 0.00 / 0.19 / 0.81 | 0.00 / 0.20 / 0.80 |
| `log_words` | BPE − LSTM | 0.00 / 0.21 / 0.78 | 0.10 / 0.66 / 0.24 | 0.00 / 0.23 / 0.77 |
| `log_words` | pool − LSTM | 0.16 / 0.67 / 0.17 | 0.72 / 0.28 / 0.00 | 0.18 / 0.67 / 0.15 |
| `sentence_length` | BPE − pool | 0.88 / 0.12 / 0.00 | 0.63 / 0.35 / 0.02 | 0.80 / 0.20 / 0.01 |
| `sentence_length` | BPE − LSTM | 0.73 / 0.27 / 0.01 | 0.55 / 0.43 / 0.02 | 0.81 / 0.19 / 0.00 |
| `sentence_length` | pool − LSTM | 0.07 / 0.56 / 0.37 | 0.13 / 0.62 / 0.24 | 0.19 / 0.62 / 0.18 |

For `em_dash_rate` and `semicolon_colon_rate`, every difference has 0.45–0.67 of its mass
inside the ROPE and no consistent sign across seeds. No difference between systems has
more than 0.95 of its mass on one side of the ROPE. `weight_differences.csv` contains all
pairs.

### Interpretation

"Practical" and "probable" refer to the ROPE rule in the method.

Practical effects, in all three seeds:

1. **Longer passages are easier for all systems** (1.00 above the ROPE everywhere). One
   standard deviation more words raises the BPE accuracy by approximately 5–6 points and
   the GloVe accuracy by approximately 4 points.
2. **Long sentences help both GloVe systems** (1.00 above the ROPE). For BPE the effect is
   positive but only probable (0.90–0.91 above the ROPE).

Probable differences between designs:

3. **Length helps BPE more than mean pooling** (0.79–0.81 above the ROPE in all seeds).
   Each extra token can switch on one more indicator, while a mean of word vectors changes
   little when more words are added. Against the LSTM, the same difference is probable for
   seeds 42 and 73 (0.77–0.78), but not for seed 17 (0.24).
4. **Long sentences probably help the GloVe systems more than BPE**, but not in every seed.
   BPE − pool has 0.80–0.88 below the ROPE for seeds 42 and 73 and 0.63 for seed 17. BPE −
   LSTM has 0.73–0.81 and 0.55. BPE indicators record which tokens occur, but not how they
   are arranged. A possible cause is that GloVe vectors carry more useful information in
   long, content-rich sentences. We did not test this cause.

No effect:

5. **Em dashes and semicolons do not make a passage easier or harder by themselves.** The
   apparent harm comes from the author model (see the method). The data also do not prove
   a practical zero: the mass inside the ROPE is 0.75–0.83 for semicolons and colons, and
   0.20–0.73 for em dashes.

Also:

6. **The systems make different errors.** All three are wrong on 47.4–48.3% of passages and
   all three are correct on only 7.7–8.2%. BPE alone is correct on 12.6–13.2%, and at least
   one system is correct on 51.7–52.6%, against 30.4% for the best single system. A
   combination of BPE and embedding systems is a good next experiment.
7. **Most errors stay inside a model family.** At the level of the six families, accuracy
   is 68.1% for BPE, 60.8% for pooling and 66.5% for the LSTM (seed 42). GLM and Kimi
   passages are mostly attributed to Claude (BPE: 0.58 and 0.54 of their passages; LSTM:
   0.54 and 0.55), and Qwen passages to GPT (0.54 and 0.50). The Claude, GLM and Kimi
   families all use many em dashes. The confusion matrices are in `artifacts/confusion/`.

### Limits

- Observations are not independent: the same passage occurs once per system, and the
  passages of one prompt share content. The posterior intervals are therefore too narrow,
  and the ROPE masses are more decisive than they should be. A hierarchical model with
  prompt and passage effects would correct this.
- The BPE system is one Part 1 run (seed 42). Its seed variance is not in this analysis.
- The analysis uses four features. A property that is not in the model can change the
  weights of correlated features. For example, in a run that also included Flesch reading
  ease (correlation −0.48 with sentence length), the LSTM weight for sentence length was
  +0.24 instead of +0.19 for seed 42.
- The features describe the passage only. Properties of the prompt, for example the topic
  or the requested length, are not included.

### Reproduce

```sh
E=artifacts/error-analysis
F="log_words sentence_length em_dash_rate semicolon_colon_rate"
for seed in 42 17 73; do
  uv run python scripts/embedding_sweep.py --pooling-grid --vectors glove300 --output $E \
    --epochs 200 --setting glove300-mlp128-lr0.003-drop0 --save-predictions --seed $seed
  uv run python scripts/embedding_sweep.py --encoder lstm --device mps --vectors glove300 \
    --output $E --lstm-reference glove300-mean-lr0.003 --epochs 25 --setting word-dropout \
    --save-predictions --seed $seed
  uv run ntl-error-analysis --control-label --features $F \
    --output $E/four-features/label-controlled-seed-$seed \
    --system bpe-indicator=artifacts/rope/log-linear-token-indicator \
    --system glove-pool-mlp=$E/pooling-mlp/seed-$seed/glove300-mlp128-lr0.003-drop0-predictions.parquet \
    --system glove-lstm=$E/lstm-followup/glove300-mean-lr0.003/seed-$seed/word-dropout-predictions.parquet
done
```

In zsh, write `${=F}` instead of `$F`. Remove `--control-label` for the analysis without
label control (`$E/four-features/uncontrolled-seed-42`). The ROPE half-width is
`--rope 0.05` by default. The BPE system uses the saved fold models of the Part 1
log-linear token-indicator run in `artifacts/rope/log-linear-token-indicator/`. Each report
contains `weights.csv`, `weight_differences.csv`, `weights.png`, `binned_accuracy.csv` and
`.png` (accuracy per feature quartile), `per_label_accuracy.csv`, `agreement.csv`,
`features.csv`, `feature_correlations.csv` and `summary.json`.

The confusion matrices are drawn from the same saved predictions, without retraining:

```sh
uv run python scripts/confusion_matrices.py \
  --system bpe-indicator=artifacts/rope/log-linear-token-indicator \
  --system glove-pool-mlp=$E/pooling-mlp/seed-42/glove300-mlp128-lr0.003-drop0-predictions.parquet \
  --system glove-lstm=$E/lstm-followup/glove300-mean-lr0.003/seed-42/word-dropout-predictions.parquet
```

## Sources

- Benavoli, Corani, Demšar and Zaffalon (2017). Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. Journal of Machine Learning Research 18.
- Nadeau and Bengio (2003). Inference for the generalization error. Machine Learning 52. This is the source of the rho = 1/n variance correction.
