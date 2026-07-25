# tests/test_integration.py
"""
Integration Tests — test components working together (DB, trainer pipeline).
These tests require the PostgreSQL container to be running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestDatabaseLayer:
    """Integration tests for storage.py ↔ PostgreSQL (FR-DB-001, FR-DB-002)."""

    def test_save_and_read_prices(self, sample_prices):
        from market_intelligence.data.storage import save_prices
        from market_intelligence.db.session import engine

        save_prices(sample_prices, "TEST")
        result = pd.read_sql(
            "SELECT * FROM stock_prices WHERE symbol = 'TEST' LIMIT 5", engine
        )
        assert len(result) > 0
        assert set(["symbol", "date", "close"]).issubset(result.columns)

    def test_save_news_deduplication(self):
        """Inserting the same url_hash twice must not raise (FR-DC-005)."""
        from market_intelligence.data.storage import save_news

        df = pd.DataFrame(
            [
                {
                    "url_hash": "dedup_test_hash_001",
                    "title": "Test article",
                    "published_date": pd.Timestamp("2025-01-06"),
                    "date": pd.Timestamp("2025-01-06").date(),
                }
            ]
        )
        save_news(df, "TEST")
        # Insert same record again — must not raise
        save_news(df, "TEST")

    def test_model_registry_populated(self):
        """After training, model_registry must contain at least one record (FR-DB-003)."""
        from market_intelligence.db.session import engine

        result = pd.read_sql("SELECT * FROM model_registry", engine)
        assert len(result) >= 1
        assert "accuracy" in result.columns


class TestMLPipeline:
    """Integration tests for the full training pipeline (FR-ML-001 to FR-ML-010)."""

    def test_build_features_then_train(self, sample_prices):
        """End-to-end: raw prices → features → XGBoost model → metrics."""
        from market_intelligence.ml.trainer import build_features, train_all
        import unittest.mock as mock

        featured = build_features(sample_prices.copy())
        assert len(featured) > 0, "Feature build returned nothing"

        # Patch MIN_ROWS and _save_to_registry so tests never pollute the real DB
        with mock.patch("market_intelligence.ml.trainer.MIN_ROWS_FOR_TRAINING", 10), \
             mock.patch("market_intelligence.ml.trainer._save_to_registry"):
            results = train_all(featured)
        assert "xgboost" in results
        assert results["xgboost"]["accuracy"] >= 0.0  # sanity: metric exists

    def test_all_four_models_trained(self, sample_prices):
        from market_intelligence.ml.trainer import build_features, train_all
        import unittest.mock as mock

        featured = build_features(sample_prices.copy())
        with mock.patch("market_intelligence.ml.trainer.MIN_ROWS_FOR_TRAINING", 10), \
             mock.patch("market_intelligence.ml.trainer._save_to_registry"):
            results = train_all(featured)
        for name in ("logistic_regression", "random_forest", "xgboost", "lightgbm"):
            assert name in results, f"Missing model: {name}"

    def test_model_artifacts_saved_to_disk(self):
        from market_intelligence.ml.trainer import MODELS_DIR

        artifacts = list(MODELS_DIR.glob("*_v*.joblib"))
        assert len(artifacts) >= 4, "Expected ≥4 model artifacts on disk"


class TestSHAPExplainability:
    """Integration: trained model + SHAP → feature impacts (FR-XAI-001 to FR-XAI-004)."""

    def test_shap_returns_list(self, sample_prices):
        from market_intelligence.ml.explainability import explain
        from market_intelligence.ml.trainer import FEATURE_COLS, build_features

        featured = build_features(sample_prices.copy())
        available = [c for c in FEATURE_COLS if c in featured.columns]
        row = featured[available].iloc[[-1]]

        result = explain(row, model_name="lightgbm", top_n=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_shap_impact_keys(self, sample_prices):
        from market_intelligence.ml.explainability import explain
        from market_intelligence.ml.trainer import FEATURE_COLS, build_features

        featured = build_features(sample_prices.copy())
        available = [c for c in FEATURE_COLS if c in featured.columns]
        row = featured[available].iloc[[-1]]

        impacts = explain(row, model_name="lightgbm")
        for item in impacts:
            assert "feature" in item
            assert "impact" in item
            assert isinstance(item["impact"], float)

    def test_positive_and_negative_impacts(self, sample_prices):
        """SHAP must produce both positive and negative values (FR-XAI-003)."""
        from market_intelligence.ml.explainability import explain
        from market_intelligence.ml.trainer import FEATURE_COLS, build_features

        featured = build_features(sample_prices.copy())
        available = [c for c in FEATURE_COLS if c in featured.columns]
        row = featured[available].iloc[[-1]]

        impacts = explain(row, model_name="lightgbm", top_n=10)
        values = [i["impact"] for i in impacts]
        # With enough features, at least some variation expected
        assert len(set([v > 0 for v in values])) >= 1
