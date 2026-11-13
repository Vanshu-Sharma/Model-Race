# Assignment 8 – Model Race: Results

## Class Balance Analysis

| Label | Count | % |
|---|---|---|
| positive | 250 | 50.0% |
| negative | 250 | 50.0% |
| **Total** | **500** | **100%** |

**Balance note:** The dataset is exactly balanced (50 % / 50 %). With a balanced dataset, accuracy is a reliable metric because a majority-class baseline achieves only 50 %, leaving no room for it to mask poor recall on the minority class. In imbalanced datasets (e.g. 90 % positive), a trivial 'always predict positive' classifier would reach 90 % accuracy while completely failing on negative samples – which is why F1-score should be preferred as the primary metric in those cases.

## Evaluation Results

Split: 80 % train / 20 % test → 400 training reviews, 100 test reviews (identical across all models)
Random state: 42 (reproducible)

| Model | Accuracy | Precision | Recall | F1 | Cost (USD) | Time/review (ms) |
|---|---|---|---|---|---|---|
| Scikit-learn (TF-IDF + LR) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | $0.000000 | 0.00 |
| Cloud LLM – gemini/gemini-3.1-flash-lite | 1.0000 | 1.0000 | 1.0000 | 1.0000 | $0.000461 | 8407.23 |
| Ollama (tinyllama) | N/A | N/A | N/A | N/A | N/A | N/A |

## Notes & Assumptions

### Pricing
- **Gemini Flash-Lite / gemini-3.1-flash-lite:** $0.075 input / $0.30 output per 1 M tokens (Google AI published list price, Nov 2026).
- **Groq (openai/gpt-oss-safeguard-20b):** $0.20 input / $0.60 output per 1 M tokens (Groq published list price).
- **Ollama:** $0.00 (local inference, no API cost).
- **Scikit-learn:** $0.00 (local CPU inference).
- Actual token counts are taken from API-reported usage metadata, not estimated.
- If token counts returned 0 (some free-tier endpoints omit usage), cost is reported as $0.00.

### Caching
- Cloud and Ollama predictions are cached to `cache/` as JSON files.
- Cache keys: SHA-256 of `{prompt_version}|{provider}|{model}|{review_text}`.
- Cached responses have 0 ms network time (excluded from average).
- Current prompt version: `v1`

### Time measurement
- Scikit-learn: wall-clock time for batch `predict()` averaged over 100 reviews.
- Cloud / Ollama: wall-clock time per individual API call (cached calls excluded from average).

## Recommendation

**Best by F1:** Scikit-learn (TF-IDF + LR) (F1 = 1.0000)

### Model comparison
- **Scikit-learn (TF-IDF + LR):** Accuracy=1.0000, F1=1.0000, Cost/100 reviews=$0.000000, Avg=0.00 ms/review
- **Cloud LLM – gemini/gemini-3.1-flash-lite:** Accuracy=1.0000, F1=1.0000, Cost/100 reviews=$0.000461, Avg=8407.23 ms/review

### Shipping recommendation

| Criterion | Recommended model |
|---|---|
| Highest accuracy / F1 | See best-by-F1 above |
| Lowest cost | Scikit-learn ($0 forever) |
| Lowest latency | Scikit-learn (sub-millisecond) |
| No internet dependency | Scikit-learn or Ollama |

**Production recommendation:** Scikit-learn (TF-IDF + Logistic Regression) for most production use cases because it has zero API cost, sub-millisecond latency, requires no internet, and delivers competitive accuracy. Use a cloud LLM only if the input distribution drifts significantly from the training domain (e.g. multilingual, mixed topics) where the in-context reasoning of an LLM adds measurable F1 improvement that justifies the per-token cost.

## Per-model notes

- **Scikit-learn (TF-IDF + LR):** No API cost. Inference time is batch-averaged.
- **Cloud LLM – gemini/gemini-3.1-flash-lite:** Provider: gemini. 1/100 from cache. Unknown fallback: 0.
- **Ollama (tinyllama):** Skipped: Ollama not available. Ollama is not running. To install and start:
  Windows  : Download from https://ollama.com/download/windows
  macOS    : brew install ollama && ollama serve
  Linux    : curl -fsSL https://ollama.com/install.sh | sh && ollama serve
  Then pull model: ollama pull tinyllama
Set OLLAMA_HOST if running on a non-default port.
