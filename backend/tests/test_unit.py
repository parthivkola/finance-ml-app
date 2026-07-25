# tests/test_unit.py
"""
Unit Tests — test individual functions in isolation, with mocks where needed.
Covers: data pipeline, NLP aggregator, feature builder, merger.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

# ──────────────────── Data Collection ─────────────────────────────────────────

class TestAlphaVantage:
    """Unit tests for AlphaVantage deduplication logic (FR-DC-005)."""

    def test_url_hash_is_deterministic(self):
        url = "https://example.com/news/apple-earnings"
        h1 = hashlib.sha256(url.encode()).hexdigest()
        h2 = hashlib.sha256(url.encode()).hexdigest()
        assert h1 == h2

    def test_url_hash_differs_for_different_urls(self):
        h1 = hashlib.sha256(b"https://a.com/1").hexdigest()
        h2 = hashlib.sha256(b"https://a.com/2").hexdigest()
        assert h1 != h2


# ──────────────────── Data Merger ─────────────────────────────────────────────

class TestMerger:
    """Unit tests for weekend alignment and merge logic (FR-DP-002)."""

    def test_merge_produces_correct_shape(self, sample_prices):
        from market_intelligence.data.merger import align_and_merge

        news_df = pd.DataFrame(
            {
                "url_hash": ["abc"],
                "title": ["Good news"],
                "published_date": [pd.Timestamp("2025-01-06")],
                "date": [pd.Timestamp("2025-01-06").date()],
            }
        )
        merged = align_and_merge(sample_prices.copy(), news_df)
        assert isinstance(merged, pd.DataFrame)
        assert len(merged) >= len(sample_prices)

    def test_weekend_saturday_aligned_to_monday(self):
        """Saturday news (weekday==5) should be pushed to Monday (+2 days)."""
        import pandas as pd

        saturday = pd.Timestamp("2025-01-04").date()  # a Saturday
        assert saturday.weekday() == 5
        aligned = saturday + pd.Timedelta(days=2)
        assert aligned.weekday() == 0  # Monday

    def test_weekend_sunday_aligned_to_monday(self):
        sunday = pd.Timestamp("2025-01-05").date()  # a Sunday
        assert sunday.weekday() == 6
        aligned = sunday + pd.Timedelta(days=1)
        assert aligned.weekday() == 0


# ──────────────────── Feature Engineering ─────────────────────────────────────

class TestFeatureBuilder:
    """Unit tests for technical indicator computation (FR-TI-001 to FR-TI-008)."""

    @pytest.fixture(autouse=True)
    def built_df(self, sample_prices):
        from market_intelligence.ml.trainer import build_features
        self.df = build_features(sample_prices.copy())

    def test_output_is_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_no_nulls_in_output(self):
        assert self.df.isnull().sum().sum() == 0

    def test_rsi_column_present(self):
        assert "RSI" in self.df.columns

    def test_rsi_in_valid_range(self):
        assert self.df["RSI"].between(0, 100).all()

    def test_sma_columns_present(self):
        assert "SMA_20" in self.df.columns
        assert "SMA_50" in self.df.columns

    def test_macd_columns_present(self):
        assert "MACD" in self.df.columns
        assert "MACD_signal" in self.df.columns

    def test_bollinger_bands_present(self):
        assert all(c in self.df.columns for c in ["BB_high", "BB_low", "BB_width"])

    def test_target_is_binary(self):
        assert set(self.df["target"].unique()).issubset({0, 1})

    def test_daily_return_present(self):
        assert "daily_return" in self.df.columns

    def test_volatility_present(self):
        assert "volatility_20" in self.df.columns


# ──────────────────── NLP Aggregator ──────────────────────────────────────────

class TestNLPAggregator:
    """Unit tests for daily sentiment aggregation (FR-NLP-006)."""

    def test_positive_sentiment_positive_score(self):
        from market_intelligence.nlp.aggregator import aggregate_daily_sentiment

        df = pd.DataFrame(
            {
                "aligned_date": ["2025-01-06", "2025-01-06"],
                "sentiment_label": ["positive", "positive"],
                "sentiment_score": [0.9, 0.8],
            }
        )
        result = aggregate_daily_sentiment(df)
        assert result["weighted_sentiment"].iloc[0] > 0

    def test_negative_sentiment_negative_score(self):
        from market_intelligence.nlp.aggregator import aggregate_daily_sentiment

        df = pd.DataFrame(
            {
                "aligned_date": ["2025-01-06"],
                "sentiment_label": ["negative"],
                "sentiment_score": [0.95],
            }
        )
        result = aggregate_daily_sentiment(df)
        assert result["weighted_sentiment"].iloc[0] < 0

    def test_neutral_sentiment_zero(self):
        from market_intelligence.nlp.aggregator import aggregate_daily_sentiment

        df = pd.DataFrame(
            {
                "aligned_date": ["2025-01-06"],
                "sentiment_label": ["neutral"],
                "sentiment_score": [0.6],
            }
        )
        result = aggregate_daily_sentiment(df)
        assert result["weighted_sentiment"].iloc[0] == 0.0


# ──────────────────── VADER Analyzer ──────────────────────────────────────────

class TestVADER:
    """Unit tests for VADER sentiment scoring (FR-NLP-003)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from market_intelligence.nlp.vader import VaderAnalyzer
        self.analyzer = VaderAnalyzer()

    def test_positive_text(self):
        result = self.analyzer.analyze("Stock surged to record highs on strong earnings!")
        assert result["label"] == "positive"

    def test_negative_text(self):
        result = self.analyzer.analyze("This is the worst, most terrible day ever, I hate it!")
        assert result["label"] == "negative"

    def test_score_in_range(self):
        result = self.analyzer.analyze("The market opened.")
        assert 0.0 <= result["score"] <= 1.0

    def test_keys_present(self):
        result = self.analyzer.analyze("hello")
        assert set(result.keys()) == {"label", "score", "raw_scores"}
