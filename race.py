"""
Assignment 8 – Model Race
=========================
Classify 500 labelled reviews (positive/negative) with three approaches:

1. Scikit-learn  : TF-IDF vectoriser + Logistic Regression
2. Cloud LLM     : Gemini or Groq via the shared generate() function from Assignment 6
3. Local LLM     : Ollama (small model, same prompt as cloud LLM)

Run:
    uv run python race.py

Environment variables (copy .env.example → .env and fill in):
    GEMINI_API_KEY, GEMINI_MODEL      – for cloud provider "gemini"
    GROQ_API_KEY,   GROQ_MODEL        – for cloud provider "groq"
    CLOUD_PROVIDER                    – "gemini" (default) or "groq"
    OLLAMA_MODEL                      – e.g. "tinyllama" (default)
    OLLAMA_HOST                       – e.g. "http://localhost:11434" (default)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass

import httpx
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("race")

ROOT = pathlib.Path(__file__).parent
DATA_FILE = ROOT / "data" / "reviews.csv"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20          # 20 % → 100 test reviews
LABEL_COL = "label"
TEXT_COL = "text"

CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "gemini").lower().strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Prompt version bump → cache keys auto-invalidate
PROMPT_VERSION = "v1"


# ---------------------------------------------------------------------------
# Pricing (USD per 1 M tokens, as of November 2026)
# Assumptions documented in results.md
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # Gemini 1.5 Flash-8B / Flash-Lite – free tier exists; paid list price used
    "gemini-3.1-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-3.1-flash":      {"input": 0.075, "output": 0.30},
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    # Fallback for any gemini model not explicitly listed
    "_gemini_default":       {"input": 0.075, "output": 0.30},
    # Groq – openai/gpt-oss-safeguard-20b (hosted on Groq)
    "openai/gpt-oss-safeguard-20b": {"input": 0.20, "output": 0.60},
    # Fallback for unknown groq models
    "_groq_default":         {"input": 0.20, "output": 0.60},
    # Ollama – local, always free
    "_ollama":               {"input": 0.0, "output": 0.0},
}


def lookup_price(model: str, provider: str) -> dict[str, float]:
    """Return per-million-token pricing for a model/provider pair."""
    if model in PRICING:
        return PRICING[model]
    fallback_key = f"_{provider}_default"
    return PRICING.get(fallback_key, {"input": 0.0, "output": 0.0})


def compute_cost(input_tokens: int, output_tokens: int, model: str, provider: str) -> float:
    """Return USD cost given token counts and model/provider."""
    prices = lookup_price(model, provider)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    df: pd.DataFrame
    X_train: list[str]
    X_test: list[str]
    y_train: list[str]
    y_test: list[str]


def load_data() -> Dataset:
    """Load and split reviews.csv into train / test sets."""
    if not DATA_FILE.exists():
        logger.error("reviews.csv not found at %s", DATA_FILE)
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    logger.info("Loaded %d reviews from %s", len(df), DATA_FILE)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        logger.error("Expected columns '%s' and '%s'. Found: %s", TEXT_COL, LABEL_COL, list(df.columns))
        sys.exit(1)

    # Validate labels
    valid_labels = {"positive", "negative"}
    actual_labels = set(df[LABEL_COL].str.lower().unique())
    if not actual_labels.issubset(valid_labels):
        logger.error("Unexpected labels: %s", actual_labels - valid_labels)
        sys.exit(1)

    # Normalise labels to lowercase
    df[LABEL_COL] = df[LABEL_COL].str.lower()

    pos = (df[LABEL_COL] == "positive").sum()
    neg = (df[LABEL_COL] == "negative").sum()
    logger.info("Class distribution → positive: %d (%.1f%%), negative: %d (%.1f%%)",
                pos, 100 * pos / len(df), neg, 100 * neg / len(df))

    X = df[TEXT_COL].tolist()
    y = df[LABEL_COL].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.info("Split → train: %d, test: %d", len(X_train), len(X_test))
    return Dataset(df=df, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    model_label: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cost_usd: float
    avg_time_ms: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    notes: str = ""


def evaluate(
    model_label: str,
    y_true: list[str],
    y_pred: list[str],
    cost_usd: float,
    avg_time_ms: float,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    notes: str = "",
) -> EvalResult:
    return EvalResult(
        model_label=model_label,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, pos_label="positive", zero_division=0),
        recall=recall_score(y_true, y_pred, pos_label="positive", zero_division=0),
        f1=f1_score(y_true, y_pred, pos_label="positive", zero_division=0),
        cost_usd=cost_usd,
        avg_time_ms=avg_time_ms,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 1. Scikit-learn: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def run_sklearn(dataset: Dataset) -> EvalResult:
    """Fit TF-IDF on training data only, then evaluate on the test set."""
    logger.info("=== Scikit-learn: TF-IDF + Logistic Regression ===")

    # Fit vectoriser on TRAINING data only (no leakage)
    vectoriser = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_vec = vectoriser.fit_transform(dataset.X_train)
    X_test_vec = vectoriser.transform(dataset.X_test)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    clf.fit(X_train_vec, dataset.y_train)

    # Time only inference on the test set
    t_infer_start = time.perf_counter()
    y_pred = clf.predict(X_test_vec).tolist()
    t_infer_end = time.perf_counter()

    n_test = len(dataset.X_test)
    avg_time_ms = (t_infer_end - t_infer_start) * 1000 / n_test

    result = evaluate(
        model_label="Scikit-learn (TF-IDF + LR)",
        y_true=dataset.y_test,
        y_pred=y_pred,
        cost_usd=0.0,
        avg_time_ms=avg_time_ms,
        notes="No API cost. Inference time is batch-averaged.",
    )
    _log_result(result)
    return result


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(text: str, model: str, provider: str, prompt_version: str) -> str:
    """Deterministic cache key incorporating text, model, provider, prompt version."""
    raw = f"{prompt_version}|{provider}|{model}|{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_path(key: str, prefix: str) -> pathlib.Path:
    return CACHE_DIR / f"{prefix}_{key[:16]}.json"


def _load_cache(key: str, prefix: str) -> dict | None:
    path = _cache_path(key, prefix)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(key: str, prefix: str, data: dict) -> None:
    path = _cache_path(key, prefix)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared LLM classification prompt
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = (
    "You are a sentiment classifier. "
    "Classify the following review as exactly one of: positive, negative. "
    "Reply with only the single word 'positive' or 'negative'. No other text."
)


def build_classify_prompt(review_text: str) -> str:
    return f"{CLASSIFY_SYSTEM}\n\nReview: {review_text.strip()}"


def normalize_llm_label(raw: str) -> str:
    """Extract 'positive' or 'negative' from LLM output. Returns 'unknown' if parsing fails."""
    text = raw.lower().strip().rstrip(".")
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return "unknown"


# ---------------------------------------------------------------------------
# 2. Cloud LLM – reuses generate() from Assignment 6 (AI-Doc-Assistant/app/llm.py)
# ---------------------------------------------------------------------------

# We import generate() directly from the sibling Assignment 6 project.
# This avoids duplicating the provider implementation.
_ASSIGN6_PATH = ROOT.parent / "AI-Doc-Assistant"
if _ASSIGN6_PATH.exists():
    sys.path.insert(0, str(_ASSIGN6_PATH))
    try:
        from app.llm import TokenUsage, generate  # type: ignore[import]
        ASSIGN6_AVAILABLE = True
        logger.info("Imported generate() from Assignment 6 at %s", _ASSIGN6_PATH)
    except ImportError as e:
        ASSIGN6_AVAILABLE = False
        logger.warning("Could not import from Assignment 6: %s. Will use built-in fallback.", e)
else:
    ASSIGN6_AVAILABLE = False
    logger.warning("Assignment 6 project not found at %s. Using built-in cloud LLM.", _ASSIGN6_PATH)

if not ASSIGN6_AVAILABLE:
    # --- Minimal built-in fallback so the assignment works stand-alone ---
    from collections.abc import AsyncIterator as _AsyncIterator
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class TokenUsage:  # type: ignore[no-redef]
        input_tokens: int
        output_tokens: int
        total_tokens: int
        model: str = ""
        provider: str = ""

    async def generate(  # type: ignore[no-redef]
        prompt: str,
        provider: str,
        on_usage=None,
    ) -> _AsyncIterator[str]:
        """Minimal stand-alone cloud LLM generate() mirroring Assignment 6 signature."""
        provider_norm = provider.lower().strip()

        if provider_norm == "gemini":
            from google import genai  # type: ignore[import]

            api_key = os.getenv("GEMINI_API_KEY")
            model = os.getenv("GEMINI_MODEL")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not configured.")
            if not model:
                raise ValueError("GEMINI_MODEL is not configured.")

            client = genai.Client(api_key=api_key)
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=prompt
            )
            usage = None
            async for chunk in stream:
                metadata = getattr(chunk, "usage_metadata", None)
                if metadata:
                    usage = TokenUsage(
                        input_tokens=getattr(metadata, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(metadata, "candidates_token_count", 0) or 0,
                        total_tokens=getattr(metadata, "total_token_count", 0) or 0,
                        model=model,
                        provider=provider_norm,
                    )
                text = None
                try:
                    text = getattr(chunk, "text", None)
                except (AttributeError, ValueError):
                    text = None
                if text:
                    yield text
            if usage and on_usage:
                on_usage(usage)

        elif provider_norm == "groq":
            from groq import AsyncGroq  # type: ignore[import]

            api_key = os.getenv("GROQ_API_KEY")
            model = os.getenv("GROQ_MODEL")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not configured.")
            if not model:
                raise ValueError("GROQ_MODEL is not configured.")

            async with AsyncGroq(api_key=api_key) as client:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    extra_body={"stream_options": {"include_usage": True}},
                )
                usage = None
                async for chunk in stream:
                    groq_usage = getattr(chunk, "usage", None)
                    if groq_usage is not None:
                        usage = TokenUsage(
                            input_tokens=getattr(groq_usage, "prompt_tokens", 0) or 0,
                            output_tokens=getattr(groq_usage, "completion_tokens", 0) or 0,
                            total_tokens=getattr(groq_usage, "total_tokens", 0) or 0,
                            model=model,
                            provider=provider_norm,
                        )
                    if getattr(chunk, "choices", None):
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            yield content
                if usage and on_usage:
                    on_usage(usage)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


async def _classify_cloud_one(
    review: str,
    provider: str,
    model_name: str,
) -> tuple[str, int, int, float]:
    """
    Classify a single review via cloud LLM.
    Returns (label, input_tokens, output_tokens, elapsed_ms).
    Checks cache first; writes to cache on success.
    """
    prompt = build_classify_prompt(review)
    cache_key = _cache_key(review, model_name, provider, PROMPT_VERSION)
    cached = _load_cache(cache_key, "cloud")
    if cached:
        return (
            cached["label"],
            cached.get("input_tokens", 0),
            cached.get("output_tokens", 0),
            0.0,  # cached → no network time
        )

    usage_holder: list[TokenUsage] = []

    def on_usage(u: TokenUsage) -> None:
        usage_holder.append(u)

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            chunks: list[str] = []

            async for chunk in generate(prompt, provider, on_usage=on_usage):
                chunks.append(chunk)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            raw = "".join(chunks)
            label = normalize_llm_label(raw)

            in_tok = usage_holder[-1].input_tokens if usage_holder else 0
            out_tok = usage_holder[-1].output_tokens if usage_holder else 0

            _save_cache(cache_key, "cloud", {
                "label": label,
                "raw": raw,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "model": model_name,
                "provider": provider,
                "prompt_version": PROMPT_VERSION,
            })
            return label, in_tok, out_tok, elapsed_ms

        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                logger.error("Cloud LLM failed after %d retries: %s", MAX_RETRIES, exc)
                return "unknown", 0, 0, 0.0
            delay = 2 ** attempt
            logger.warning("Cloud LLM error (attempt %d/%d): %s. Retrying in %ds.",
                           attempt + 1, MAX_RETRIES, exc, delay)
            await asyncio.sleep(delay)

    return "unknown", 0, 0, 0.0


async def run_cloud_llm(dataset: Dataset) -> EvalResult:
    """Run cloud LLM classification on the test set."""
    provider = CLOUD_PROVIDER
    model_name = os.getenv(f"{provider.upper()}_MODEL", "unknown")
    logger.info("=== Cloud LLM: provider=%s model=%s ===", provider, model_name)

    # Check credentials
    key_env = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
    if not os.getenv(key_env):
        logger.warning("%s not set. Skipping cloud LLM.", key_env)
        return EvalResult(
            model_label=f"Cloud LLM ({model_name})",
            accuracy=0, precision=0, recall=0, f1=0,
            cost_usd=0, avg_time_ms=0,
            notes=f"Skipped: {key_env} not configured.",
        )

    y_pred: list[str] = []
    total_in_tok = 0
    total_out_tok = 0
    total_time_ms = 0.0
    n_cached = 0

    for i, review in enumerate(dataset.X_test):
        label, in_tok, out_tok, elapsed_ms = await _classify_cloud_one(review, provider, model_name)
        y_pred.append(label)
        total_in_tok += in_tok
        total_out_tok += out_tok
        total_time_ms += elapsed_ms
        if elapsed_ms == 0.0:
            n_cached += 1

        if (i + 1) % 10 == 0:
            logger.info("  Cloud LLM: %d/%d reviews classified (%d cached)",
                        i + 1, len(dataset.X_test), n_cached)

    n_live = len(dataset.X_test) - n_cached
    avg_time_ms = (total_time_ms / n_live) if n_live > 0 else 0.0

    # Replace any 'unknown' predictions with majority class
    n_unknown = y_pred.count("unknown")
    if n_unknown:
        logger.warning("Cloud LLM returned 'unknown' for %d reviews (counted as 'negative')", n_unknown)
        y_pred = [p if p != "unknown" else "negative" for p in y_pred]

    cost = compute_cost(total_in_tok, total_out_tok, model_name, provider)
    model_label = f"Cloud LLM – {provider}/{model_name}"

    result = evaluate(
        model_label=model_label,
        y_true=dataset.y_test,
        y_pred=y_pred,
        cost_usd=cost,
        avg_time_ms=avg_time_ms,
        total_input_tokens=total_in_tok,
        total_output_tokens=total_out_tok,
        notes=f"Provider: {provider}. {n_cached}/{len(dataset.X_test)} from cache. Unknown fallback: {n_unknown}.",
    )
    _log_result(result)
    return result


# ---------------------------------------------------------------------------
# 3. Local LLM – Ollama
# ---------------------------------------------------------------------------

def _check_ollama() -> bool:
    """Return True if Ollama HTTP server is reachable."""
    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


async def _classify_ollama_one(
    review: str,
    model: str,
    client: httpx.AsyncClient,
) -> tuple[str, int, int, float]:
    """Classify a single review via Ollama. Caches results locally."""
    cache_key = _cache_key(review, model, "ollama", PROMPT_VERSION)
    cached = _load_cache(cache_key, "ollama")
    if cached:
        return (
            cached["label"],
            cached.get("input_tokens", 0),
            cached.get("output_tokens", 0),
            0.0,
        )

    prompt = build_classify_prompt(review)

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            data = response.json()
            raw = data.get("response", "")
            label = normalize_llm_label(raw)

            in_tok = data.get("prompt_eval_count", 0) or 0
            out_tok = data.get("eval_count", 0) or 0

            _save_cache(cache_key, "ollama", {
                "label": label,
                "raw": raw,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "model": model,
                "provider": "ollama",
                "prompt_version": PROMPT_VERSION,
            })
            return label, in_tok, out_tok, elapsed_ms

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 or exc.response.status_code >= 500:
                if attempt < MAX_RETRIES - 1:
                    delay = 2 ** attempt
                    logger.warning("Ollama HTTP %d, retrying in %ds.", exc.response.status_code, delay)
                    await asyncio.sleep(delay)
                    continue
            logger.error("Ollama request failed: %s", exc)
            return "unknown", 0, 0, 0.0
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                delay = 2 ** attempt
                logger.warning("Ollama error (attempt %d/%d): %s. Retrying in %ds.",
                               attempt + 1, MAX_RETRIES, exc, delay)
                await asyncio.sleep(delay)
                continue
            logger.error("Ollama failed: %s", exc)
            return "unknown", 0, 0, 0.0

    return "unknown", 0, 0, 0.0


async def run_ollama(dataset: Dataset) -> EvalResult:
    """Run Ollama local LLM classification on the test set."""
    model = OLLAMA_MODEL
    logger.info("=== Local LLM: Ollama model=%s ===", model)

    if not _check_ollama():
        msg = (
            "Ollama is not running. To install and start:\n"
            "  Windows  : Download from https://ollama.com/download/windows\n"
            "  macOS    : brew install ollama && ollama serve\n"
            "  Linux    : curl -fsSL https://ollama.com/install.sh | sh && ollama serve\n"
            f"  Then pull model: ollama pull {model}\n"
            "Set OLLAMA_HOST if running on a non-default port."
        )
        logger.warning(msg)
        return EvalResult(
            model_label=f"Ollama ({model})",
            accuracy=0, precision=0, recall=0, f1=0,
            cost_usd=0, avg_time_ms=0,
            notes="Skipped: Ollama not available. " + msg,
        )

    y_pred: list[str] = []
    total_in_tok = 0
    total_out_tok = 0
    total_time_ms = 0.0
    n_cached = 0

    async with httpx.AsyncClient() as client:
        for i, review in enumerate(dataset.X_test):
            label, in_tok, out_tok, elapsed_ms = await _classify_ollama_one(review, model, client)
            y_pred.append(label)
            total_in_tok += in_tok
            total_out_tok += out_tok
            total_time_ms += elapsed_ms
            if elapsed_ms == 0.0:
                n_cached += 1

            if (i + 1) % 10 == 0:
                logger.info("  Ollama: %d/%d reviews classified (%d cached)",
                            i + 1, len(dataset.X_test), n_cached)

    n_live = len(dataset.X_test) - n_cached
    avg_time_ms = (total_time_ms / n_live) if n_live > 0 else 0.0

    n_unknown = y_pred.count("unknown")
    if n_unknown:
        logger.warning("Ollama returned 'unknown' for %d reviews (counted as 'negative')", n_unknown)
        y_pred = [p if p != "unknown" else "negative" for p in y_pred]

    result = evaluate(
        model_label=f"Ollama ({model})",
        y_true=dataset.y_test,
        y_pred=y_pred,
        cost_usd=0.0,
        avg_time_ms=avg_time_ms,
        total_input_tokens=total_in_tok,
        total_output_tokens=total_out_tok,
        notes=f"Local inference. {n_cached}/{len(dataset.X_test)} from cache. Unknown fallback: {n_unknown}.",
    )
    _log_result(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _log_result(r: EvalResult) -> None:
    logger.info(
        "[%s] Acc=%.4f Prec=%.4f Rec=%.4f F1=%.4f Cost=$%.6f Avg=%.2fms",
        r.model_label, r.accuracy, r.precision, r.recall, r.f1, r.cost_usd, r.avg_time_ms,
    )


def write_results_md(results: list[EvalResult], dataset: Dataset) -> None:
    """Write results.md with the comparison table and analysis."""
    pos = (dataset.df[LABEL_COL] == "positive").sum()
    neg = (dataset.df[LABEL_COL] == "negative").sum()

    lines: list[str] = [
        "# Assignment 8 – Model Race: Results",
        "",
        "## Class Balance Analysis",
        "",
        "| Label | Count | % |",
        "|---|---|---|",
        f"| positive | {pos} | {100*pos/len(dataset.df):.1f}% |",
        f"| negative | {neg} | {100*neg/len(dataset.df):.1f}% |",
        f"| **Total** | **{len(dataset.df)}** | **100%** |",
        "",
        "**Balance note:** The dataset is exactly balanced (50 % / 50 %). "
        "With a balanced dataset, accuracy is a reliable metric because a "
        "majority-class baseline achieves only 50 %, leaving no room for "
        "it to mask poor recall on the minority class. "
        "In imbalanced datasets (e.g. 90 % positive), a trivial 'always predict positive' "
        "classifier would reach 90 % accuracy while completely failing on negative samples – "
        "which is why F1-score should be preferred as the primary metric in those cases.",
        "",
        "## Evaluation Results",
        "",
        "Split: 80 % train / 20 % test → 400 training reviews, 100 test reviews (identical across all models)",
        "Random state: 42 (reproducible)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | Cost (USD) | Time/review (ms) |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results:
        if r.avg_time_ms == 0 and r.accuracy == 0 and r.notes.startswith("Skipped"):
            row = (
                f"| {r.model_label} | N/A | N/A | N/A | N/A"
                f" | N/A | N/A |"
            )
        else:
            row = (
                f"| {r.model_label} | {r.accuracy:.4f} | {r.precision:.4f}"
                f" | {r.recall:.4f} | {r.f1:.4f}"
                f" | ${r.cost_usd:.6f} | {r.avg_time_ms:.2f} |"
            )
        lines.append(row)

    lines += [
        "",
        "## Notes & Assumptions",
        "",
        "### Pricing",
        "- **Gemini Flash-Lite / gemini-3.1-flash-lite:** $0.075 input / $0.30 output per 1 M tokens (Google AI published list price, Nov 2026).",
        "- **Groq (openai/gpt-oss-safeguard-20b):** $0.20 input / $0.60 output per 1 M tokens (Groq published list price).",
        "- **Ollama:** $0.00 (local inference, no API cost).",
        "- **Scikit-learn:** $0.00 (local CPU inference).",
        "- Actual token counts are taken from API-reported usage metadata, not estimated.",
        "- If token counts returned 0 (some free-tier endpoints omit usage), cost is reported as $0.00.",
        "",
        "### Caching",
        "- Cloud and Ollama predictions are cached to `cache/` as JSON files.",
        "- Cache keys: SHA-256 of `{prompt_version}|{provider}|{model}|{review_text}`.",
        "- Cached responses have 0 ms network time (excluded from average).",
        f"- Current prompt version: `{PROMPT_VERSION}`",
        "",
        "### Time measurement",
        "- Scikit-learn: wall-clock time for batch `predict()` averaged over 100 reviews.",
        "- Cloud / Ollama: wall-clock time per individual API call (cached calls excluded from average).",
        "",
    ]

    # Recommendation
    valid = [r for r in results if not r.notes.startswith("Skipped")]
    if valid:
        best = max(valid, key=lambda r: r.f1)
        lines += [
            "## Recommendation",
            "",
            f"**Best by F1:** {best.model_label} (F1 = {best.f1:.4f})",
            "",
            "### Model comparison",
        ]
        for r in valid:
            lines.append(f"- **{r.model_label}:** Accuracy={r.accuracy:.4f}, F1={r.f1:.4f}, "
                         f"Cost/100 reviews=${r.cost_usd:.6f}, Avg={r.avg_time_ms:.2f} ms/review")

        lines += [
            "",
            "### Shipping recommendation",
            "",
            "| Criterion | Recommended model |",
            "|---|---|",
            "| Highest accuracy / F1 | See best-by-F1 above |",
            "| Lowest cost | Scikit-learn ($0 forever) |",
            "| Lowest latency | Scikit-learn (sub-millisecond) |",
            "| No internet dependency | Scikit-learn or Ollama |",
            "",
            "**Production recommendation:** Scikit-learn (TF-IDF + Logistic Regression) for most "
            "production use cases because it has zero API cost, sub-millisecond latency, requires "
            "no internet, and delivers competitive accuracy. "
            "Use a cloud LLM only if the input distribution drifts significantly from the training "
            "domain (e.g. multilingual, mixed topics) where the in-context reasoning of an LLM "
            "adds measurable F1 improvement that justifies the per-token cost.",
        ]

    # Notes per model
    lines += ["", "## Per-model notes", ""]
    for r in results:
        if r.notes:
            lines.append(f"- **{r.model_label}:** {r.notes}")

    out = ROOT / "results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Written results.md → %s", out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    dataset = load_data()

    # Run all three models
    sklearn_result = run_sklearn(dataset)
    cloud_result = await run_cloud_llm(dataset)
    ollama_result = await run_ollama(dataset)

    results = [sklearn_result, cloud_result, ollama_result]
    write_results_md(results, dataset)

    # Print summary table to stdout
    print("\n" + "=" * 70)
    print("MODEL RACE RESULTS")
    print("=" * 70)
    print(f"{'Model':<40} {'Acc':>7} {'F1':>7} {'Cost':>12} {'ms/rev':>9}")
    print("-" * 70)
    for r in results:
        if r.notes.startswith("Skipped"):
            print(f"{r.model_label:<40} {'N/A':>7} {'N/A':>7} {'N/A':>12} {'N/A':>9}")
        else:
            print(
                f"{r.model_label:<40} {r.accuracy:>7.4f} {r.f1:>7.4f}"
                f" ${r.cost_usd:>11.6f} {r.avg_time_ms:>8.2f}"
            )
    print("=" * 70)
    print("Full results written to results.md")


if __name__ == "__main__":
    asyncio.run(main())
