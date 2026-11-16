# Expanded-data and frozen-backbone comparison

Version 13 of `prvsiyan/prvsiyans-extended-name-that-llm-dataset` was downloaded
through the public Kaggle endpoint. The manifest is in `data/raw/extended-v13`.
This downloads the same dataset identified by the requested kagglehub handle.
The raw file has 15,226 rows. The complete-block clean file is unchanged from v7.

`prepare_extended` selects quality-filter-passing rows from new, new2, and proxy_v1,
including incomplete blocks: 9,482 rows, 660 prompts. It excludes reconstructed
competition prompts, one-author reruns, and the pilot protocol. The source's filter
uses competition overlap heuristics; this is not an entirely untouched benchmark.
Provider-proxy routes and missing author cells remain potential confounds.
All comparisons use the same newly assigned prompt-disjoint folds. Do not attribute
changes versus the earlier 4,818-row results solely to training-set size: validation
composition also changes.

The frozen backbone starts with `sentence-transformers/all-MiniLM-L6-v2`, pinned to
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Masked mean pooling and L2
normalization follow its model card. This is a compact semantic encoder, not a
competitive fine-tuned attribution model. Its semantic objective may suppress the
style differences that matter here. The extractor also accepts other AutoModel
checkpoints; pooling, sequence length, and validation must suit the chosen model.
No author-label training occurs before folds. PCA is fitted separately per fold.
The shared Gaussian is compared with full-vector logistic regression, reduced-vector
logistic regression, LDA, naive centering, and matched independent/shuffle controls.

The extractor logs truncation: 1,022 of 9,482 passages exceed 256 tokens. This is a
material limitation. No claim about a fully fine-tuned backbone follows from this run.

Commands (new output paths required):

```sh
uv run python scripts/download_data.py --version 13 --split-output data/extended-v13
uv run python -m name_that_llm.research.prepare_extended
HF_HOME=models/huggingface uv run python -m name_that_llm.research.backbone \
  --data-dir data/extended-v13-expanded --output artifacts/prompt-effects/minilm-expanded.npz
uv run python -m name_that_llm.research.run --data-dir data/extended-v13-expanded \
  --features embeddings --embeddings artifacts/prompt-effects/minilm-expanded.npz \
  --embedding-provenance 'See minilm-expanded.json: pinned frozen MiniLM, mean pooling, 256 tokens' \
  --max-iter 1000 --output artifacts/prompt-effects/minilm-expanded
uv run python -m name_that_llm.research.run --data-dir data/extended-v13-expanded \
  --features char --max-iter 1000 --output artifacts/prompt-effects/char-expanded
uv run python -m name_that_llm.research.families artifacts/prompt-effects/minilm-expanded
```

Family diagnostics use explicit name-based groups: Claude Opus/Sonnet/Fable,
GPT-5.6 variants, Gemini Flash variants, GLM-5.3 variants; other labels are singleton
families. These are operational categories, not verified training genealogy.
Reports decompose exact-author errors into within-family, other-family same-provider,
and cross-provider rates, with equal weight per exact author. Provider agreement is
computed from the exact-label prediction; it is not a separately trained provider
classifier. These diagnostics do not use the true family to constrain predictions.
Hard family routing is deliberately not introduced: a wrong routing decision would
make the correct author unreachable. A hierarchical training objective would need a
separate, same-backbone ablation before adoption.

The BPE default is now 4,000 rather than 8,000. A fold-0 tokenizer was fitted on the
expanded training partition and preprocessing completed. This change does not affect
character TF-IDF or the pretrained tokenizer. Accuracy superiority of 4,000 is not
established; previous 8,000-vocabulary experimental results remain historical.

Model reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
