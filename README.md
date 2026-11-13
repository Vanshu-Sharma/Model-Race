# Model Race – Assignment 8

## Objective

Classify 500 labelled product reviews (positive / negative) using **three approaches** and compare their performance on an identical held-out test set:

| # | Approach | Method |
|---|---|---|
| 1 | **Scikit-learn** | TF-IDF vectoriser + Logistic Regression |
| 2 | **Cloud LLM** | Gemini or Groq via the shared `generate()` from Assignment 6 |
| 3 | **Local LLM** | Ollama (small model, same prompt as cloud LLM) |

---

## Project Structure

```
model-race/
├── data/
│   └── reviews.csv        # 500 labelled reviews (review_id, text, label)
├── cache/                 # Auto-created; stores cloud & Ollama prediction JSON files
├── scripts/
│   └── generate_reviews.py  # One-off script to regenerate data/reviews.csv
├── tests/
│   └── test_race.py       # Unit tests
├── race.py                # Main evaluation script
├── results.md             # Auto-generated comparison table & analysis
├── README.md              # This file
├── pyproject.toml         # Dependencies (uv)
├── uv.lock                # Locked dependency versions
├── .env.example           # Environment variable template
└── .gitignore
```

---

## Architecture

```
race.py
├── load_data()            → reads reviews.csv, 80/20 stratified split (seed=42)
│
├── run_sklearn()          → TF-IDF fit on train only → LogReg predict on test
│
├── run_cloud_llm()        → per-review async calls via generate() (Assign 6)
│   ├── _classify_cloud_one()  → check cache → call API → parse → cache
│   └── compute_cost()         → token-count × published list price
│
├── run_ollama()           → per-review async HTTP calls to Ollama /api/generate
│   └── _classify_ollama_one() → check cache → POST Ollama → parse → cache
│
└── write_results_md()     → metrics table + class-balance analysis + recommendation
```

### Assignment 6 `generate()` integration

`race.py` imports `generate()` and `TokenUsage` directly from the sibling
`AI-Doc-Assistant/app/llm.py` project (Assignment 6).  
If that path is not found, a minimal built-in fallback with the same signature is used.

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | TF-IDF, Logistic Regression, metrics |
| `pandas` | CSV loading |
| `google-genai` | Gemini provider |
| `groq` | Groq provider |
| `httpx` | Ollama HTTP client |
| `python-dotenv` | `.env` loading |

---

## Setup

### 1. Install uv (if not already)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment and install dependencies

```bash
# If on OneDrive/network drive, set this first:
$env:UV_LINK_MODE="copy"   # PowerShell
# export UV_LINK_MODE=copy  # bash

cd Model_Race
uv sync
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Install Ollama (for local LLM)

| Platform | Command |
|---|---|
| **Windows** | Download installer from https://ollama.com/download/windows |
| **macOS** | `brew install ollama` then `ollama serve` |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` then `ollama serve` |

After installing:

```bash
ollama pull tinyllama   # ~600 MB, fast on CPU
# or
ollama pull gemma3:1b   # ~815 MB
# or any other small model, then set OLLAMA_MODEL in .env
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `CLOUD_PROVIDER` | No | `gemini` | `gemini` or `groq` |
| `GEMINI_API_KEY` | Yes (if gemini) | – | Google AI API key |
| `GEMINI_MODEL` | Yes (if gemini) | – | e.g. `gemini-3.1-flash-lite` |
| `GROQ_API_KEY` | Yes (if groq) | – | Groq API key |
| `GROQ_MODEL` | Yes (if groq) | – | e.g. `openai/gpt-oss-safeguard-20b` |
| `OLLAMA_MODEL` | No | `tinyllama` | Ollama model tag |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL |

---

## Running the Evaluation

```bash
# Full evaluation (all three models)
uv run python race.py

# With copy link mode (required on OneDrive)
$env:UV_LINK_MODE="copy"; uv run python race.py

# Regenerate the dataset
uv run python scripts/generate_reviews.py
```

---

## Evaluation Methodology

### Dataset split

- **80/20 stratified split** with `random_state=42`
- 400 training reviews → 100 test reviews
- All three models evaluate on the **same 100 test reviews**
- TF-IDF vectoriser is fitted **only on training data** (no data leakage)

### Metrics (all computed on the identical test set)

| Metric | Definition |
|---|---|
| **Accuracy** | (TP + TN) / total |
| **Precision** | TP / (TP + FP) for `positive` class |
| **Recall** | TP / (TP + FN) for `positive` class |
| **F1** | Harmonic mean of precision and recall |
| **Cost (USD)** | Actual input/output tokens × published list price / 1 M |
| **Time/review** | Wall-clock ms per review (network calls); batch average for sklearn |

### Pricing assumptions

- Gemini Flash-Lite: **\$0.075 / M input, \$0.30 / M output** (Google AI list, Nov 2026)
- Groq (gpt-oss-safeguard-20b): **\$0.20 / M input, \$0.60 / M output** (Groq list, Nov 2026)
- Ollama & Scikit-learn: **\$0.00** (local)

### Caching

- Cloud and Ollama predictions are cached to `cache/` as JSON files.
- Cache key: `SHA-256(prompt_version | provider | model | review_text)`
- Re-running the script uses cached predictions without API calls.
- Change `PROMPT_VERSION` in `race.py` to invalidate the cache.

---

## Results

See **[results.md](results.md)** for the full comparison table and analysis.

---

## Tests

```bash
uv run pytest tests/ -v
```

---

## Notes

- **No data leakage:** TF-IDF is `fit()` on training data, `transform()` only on test.
- **Reproducibility:** Fixed `random_state=42` throughout.
- **Ollama skipped gracefully** if the server is not running; the script reports installation commands.
- **Unknown predictions** (if an LLM returns something other than "positive"/"negative") are logged and fallback to "negative" to avoid crashing metrics.
- **API keys** are never committed; use `.env` (listed in `.gitignore`).
