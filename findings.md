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

## Sources

- Benavoli, Corani, Demšar and Zaffalon (2017). Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. Journal of Machine Learning Research 18.
- Nadeau and Bengio (2003). Inference for the generalization error. Machine Learning 52. This is the source of the rho = 1/n variance correction.
