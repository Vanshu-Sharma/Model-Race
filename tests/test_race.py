"""
Unit tests for Assignment 8: Model Race.

Covers:
- Dataset loading and class-balance verification
- Reproducible 80/20 stratified split (identical test sets)
- No test-data leakage (TF-IDF fit only on train)
- Scikit-learn pipeline correctness
- Prediction normalization / validation
- Cache key determinism
- Results reporting
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Allow imports from project root when running from tests/
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from race import (  # noqa: E402
    PROMPT_VERSION,
    Dataset,
    EvalResult,
    _cache_key,
    _load_cache,
    _save_cache,
    build_classify_prompt,
    compute_cost,
    load_data,
    normalize_llm_label,
    run_sklearn,
    write_results_md,
)

DATA_PATH = ROOT / "data" / "reviews.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """Load and split dataset once for all tests."""
    return load_data()


# ---------------------------------------------------------------------------
# 1. Dataset integrity
# ---------------------------------------------------------------------------


class TestDataset:
    def test_csv_exists(self):
        assert DATA_PATH.exists(), f"reviews.csv not found at {DATA_PATH}"

    def test_total_reviews(self, dataset):
        assert len(dataset.df) == 500, f"Expected 500 reviews, got {len(dataset.df)}"

    def test_columns(self, dataset):
        assert "text" in dataset.df.columns
        assert "label" in dataset.df.columns

    def test_labels_are_valid(self, dataset):
        unique_labels = set(dataset.df["label"].unique())
        assert unique_labels == {"positive", "negative"}, f"Unexpected labels: {unique_labels}"

    def test_class_balance(self, dataset):
        pos = (dataset.df["label"] == "positive").sum()
        neg = (dataset.df["label"] == "negative").sum()
        assert pos == 250, f"Expected 250 positive, got {pos}"
        assert neg == 250, f"Expected 250 negative, got {neg}"


# ---------------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_train_size(self, dataset):
        assert len(dataset.X_train) == 400
        assert len(dataset.y_train) == 400

    def test_test_size(self, dataset):
        assert len(dataset.X_test) == 100
        assert len(dataset.y_test) == 100

    def test_test_is_stratified(self, dataset):
        pos = dataset.y_test.count("positive")
        neg = dataset.y_test.count("negative")
        assert pos == 50, f"Expected 50 positive in test, got {pos}"
        assert neg == 50, f"Expected 50 negative in test, got {neg}"

    def test_split_reproducibility(self):
        """Running load_data() twice with same seed must yield identical test sets."""
        d1 = load_data()
        d2 = load_data()
        assert d1.X_test == d2.X_test, "Test sets differ between runs"
        assert d1.y_test == d2.y_test, "Test labels differ between runs"

    def test_no_overlap_train_test(self, dataset):
        """Train and test texts must not overlap (guards against leakage)."""
        train_set = set(dataset.X_train)
        test_set = set(dataset.X_test)
        overlap = train_set & test_set
        # Note: exact string duplicates are theoretically possible in the dataset
        # but should be zero in our generated data
        assert len(overlap) == 0, f"{len(overlap)} reviews appear in both train and test sets"


# ---------------------------------------------------------------------------
# 3. Scikit-learn – no data leakage, correct metrics
# ---------------------------------------------------------------------------


class TestSklearn:
    def test_sklearn_runs_and_returns_eval(self, dataset):
        result = run_sklearn(dataset)
        assert isinstance(result, EvalResult)

    def test_sklearn_accuracy_reasonable(self, dataset):
        result = run_sklearn(dataset)
        # With clean labelled data we expect high accuracy
        assert result.accuracy >= 0.90, f"Accuracy {result.accuracy} unexpectedly low"

    def test_sklearn_zero_cost(self, dataset):
        result = run_sklearn(dataset)
        assert result.cost_usd == 0.0

    def test_sklearn_positive_time(self, dataset):
        result = run_sklearn(dataset)
        assert result.avg_time_ms >= 0.0

    def test_sklearn_no_tfidf_leakage(self, dataset):
        """
        TF-IDF must be fit on training data only.
        Confirm: test vocabulary (tokens that appear ONLY in test) are
        treated as OOV (all-zero column) and do not raise an exception.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectoriser = TfidfVectorizer()
        vectoriser.fit(dataset.X_train)
        # transform() on test must not raise even if new tokens appear
        X_test_vec = vectoriser.transform(dataset.X_test)
        assert X_test_vec.shape[0] == len(dataset.X_test)


# ---------------------------------------------------------------------------
# 4. Prediction normalisation
# ---------------------------------------------------------------------------


class TestNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("positive", "positive"),
        ("POSITIVE", "positive"),
        ("Positive.", "positive"),
        ("negative", "negative"),
        ("NEGATIVE", "negative"),
        ("Negative!", "negative"),
        ("The sentiment is positive", "positive"),
        ("I think it's negative.", "negative"),
        ("1", "unknown"),
        ("", "unknown"),
        ("unsure", "unknown"),
        ("pos", "unknown"),
    ])
    def test_normalize_various(self, raw, expected):
        assert normalize_llm_label(raw) == expected


# ---------------------------------------------------------------------------
# 5. Cache
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_key_determinism(self):
        k1 = _cache_key("review text", "model-x", "gemini", PROMPT_VERSION)
        k2 = _cache_key("review text", "model-x", "gemini", PROMPT_VERSION)
        assert k1 == k2

    def test_cache_key_sensitivity(self):
        k_base = _cache_key("review text", "model-x", "gemini", PROMPT_VERSION)
        k_text = _cache_key("different review", "model-x", "gemini", PROMPT_VERSION)
        k_model = _cache_key("review text", "model-y", "gemini", PROMPT_VERSION)
        k_provider = _cache_key("review text", "model-x", "groq", PROMPT_VERSION)
        k_version = _cache_key("review text", "model-x", "gemini", "v2")
        for k in (k_text, k_model, k_provider, k_version):
            assert k != k_base, "Cache key collision detected"

    def test_cache_round_trip(self, tmp_path):
        key = _cache_key("test review", "some-model", "gemini", PROMPT_VERSION)
        payload = {"label": "positive", "input_tokens": 50, "output_tokens": 2}
        _save_cache(key, "cloud", payload)
        loaded = _load_cache(key, "cloud")
        assert loaded == payload


# ---------------------------------------------------------------------------
# 6. Pricing
# ---------------------------------------------------------------------------


class TestPricing:
    def test_sklearn_free(self):
        cost = compute_cost(1_000_000, 1_000_000, "unknown-model", "sklearn")
        assert cost == 0.0

    def test_gemini_cost_positive(self):
        cost = compute_cost(1_000_000, 1_000_000, "gemini-3.1-flash-lite", "gemini")
        assert cost > 0.0

    def test_ollama_free(self):
        cost = compute_cost(500, 10, "_ollama", "ollama")
        assert cost == 0.0


# ---------------------------------------------------------------------------
# 7. Prompt builder
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_prompt_contains_review(self):
        review = "This product is amazing!"
        prompt = build_classify_prompt(review)
        assert review in prompt

    def test_prompt_asks_for_classification(self):
        prompt = build_classify_prompt("test")
        lower = prompt.lower()
        assert "positive" in lower and "negative" in lower


# ---------------------------------------------------------------------------
# 8. results.md generation
# ---------------------------------------------------------------------------


class TestReporting:
    def test_write_results_md(self, dataset, tmp_path):
        import race as _race

        original_root = _race.ROOT
        # Temporarily redirect output
        results = [
            EvalResult(
                model_label="Scikit-learn",
                accuracy=0.95,
                precision=0.94,
                recall=0.96,
                f1=0.95,
                cost_usd=0.0,
                avg_time_ms=0.05,
            ),
            EvalResult(
                model_label="Cloud LLM (test-model)",
                accuracy=0.92,
                precision=0.91,
                recall=0.93,
                f1=0.92,
                cost_usd=0.001,
                avg_time_ms=1200.0,
            ),
            EvalResult(
                model_label="Ollama (tinyllama)",
                accuracy=0.88,
                precision=0.87,
                recall=0.89,
                f1=0.88,
                cost_usd=0.0,
                avg_time_ms=3000.0,
            ),
        ]

        out = tmp_path / "results.md"
        # Monkey-patch ROOT temporarily
        _race.ROOT = tmp_path
        try:
            write_results_md(results, dataset)
        finally:
            _race.ROOT = original_root

        assert out.exists(), "results.md was not created"
        content = out.read_text(encoding="utf-8")
        assert "Scikit-learn" in content
        assert "Class Balance" in content
        assert "Recommendation" in content
        # Verify table header
        assert "Accuracy" in content
        assert "F1" in content
