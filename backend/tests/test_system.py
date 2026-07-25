# tests/test_system.py
"""
System Tests — test the full HTTP API stack end-to-end via FastAPI TestClient.
Mocks external API calls (Yahoo Finance) so tests run offline without side effects.
Covers FR-API-001 to FR-API-008.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ──────────────────── Health Endpoint ─────────────────────────────────────────

class TestHealthEndpoint:
    """FR-API-004."""

    def test_health_returns_200(self, app_client):
        resp = app_client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_json_structure(self, app_client):
        data = app_client.get("/api/v1/health").json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models_available" in data
        assert isinstance(data["models_available"], list)

    def test_swagger_docs_accessible(self, app_client):
        """FR-API-008 — Swagger UI must be reachable."""
        resp = app_client.get("/docs")
        assert resp.status_code == 200


# ──────────────────── Models Endpoint ─────────────────────────────────────────

class TestModelsEndpoint:
    """FR-API-003 — model registry with metrics."""

    def test_models_returns_200(self, app_client):
        resp = app_client.get("/api/v1/models")
        assert resp.status_code == 200

    def test_models_is_list(self, app_client):
        data = app_client.get("/api/v1/models").json()
        assert isinstance(data, list)

    def test_models_have_required_fields(self, app_client):
        data = app_client.get("/api/v1/models").json()
        if data:
            record = data[0]
            for field in ("model_name", "version", "accuracy", "f1_score", "roc_auc"):
                assert field in record, f"Missing field: {field}"


# ──────────────────── News Endpoint ───────────────────────────────────────────

class TestNewsEndpoint:
    """FR-API-002."""

    def test_news_returns_200(self, app_client):
        resp = app_client.get("/api/v1/news/AAPL")
        assert resp.status_code == 200

    def test_news_returns_list(self, app_client):
        data = app_client.get("/api/v1/news/AAPL").json()
        assert isinstance(data, list)

    def test_news_unknown_symbol_returns_empty(self, app_client):
        data = app_client.get("/api/v1/news/ZZZZ_INVALID").json()
        assert data == []


# ──────────────────── History Endpoint ────────────────────────────────────────

class TestHistoryEndpoint:
    """FR-API-005."""

    def test_history_known_symbol(self, app_client):
        resp = app_client.get("/api/v1/history/AAPL")
        assert resp.status_code == 200

    def test_history_has_ohlcv(self, app_client):
        data = app_client.get("/api/v1/history/AAPL").json()
        assert len(data) > 0
        record = data[0]
        for field in ("symbol", "date", "open", "high", "low", "close", "volume"):
            assert field in record

    def test_history_unknown_symbol_returns_404(self, app_client):
        resp = app_client.get("/api/v1/history/ZZZZ_INVALID")
        assert resp.status_code == 404


# ──────────────────── Predict Endpoint ────────────────────────────────────────

class TestPredictEndpoint:
    """FR-API-001 — with mocked Yahoo Finance to avoid live API calls in CI."""

    def _mock_feature_row(self):
        """Build a valid feature row using synthetic data without network calls."""
        from market_intelligence.ml.trainer import FEATURE_COLS, build_features
        import numpy as np

        n = 120
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        rng = np.random.default_rng(0)
        close = 200 + rng.standard_normal(n).cumsum()
        prices = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, n).astype(int),
            }
        )
        featured = build_features(prices)
        available = [c for c in FEATURE_COLS if c in featured.columns]
        return featured[available].iloc[[-1]]

    def test_predict_invalid_model_returns_422(self, app_client):
        resp = app_client.post(
            "/api/v1/predict",
            json={"symbol": "AAPL", "model_name": "nonexistent_model"},
        )
        assert resp.status_code == 422

    def test_predict_valid_model_returns_200(self, app_client):
        """Mocks _get_feature_row so no live Yahoo Finance call is made."""
        feature_row = self._mock_feature_row()

        with patch(
            "market_intelligence.api.predict._get_feature_row",
            return_value=(feature_row, True),
        ):
            resp = app_client.post(
                "/api/v1/predict",
                json={"symbol": "AAPL", "model_name": "xgboost"},
            )
        assert resp.status_code == 200

    def test_predict_response_structure(self, app_client):
        feature_row = self._mock_feature_row()

        with patch(
            "market_intelligence.api.predict._get_feature_row",
            return_value=(feature_row, True),
        ):
            data = app_client.post(
                "/api/v1/predict",
                json={"symbol": "AAPL", "model_name": "xgboost"},
            ).json()

        assert "symbol" in data
        assert "prediction" in data
        assert data["prediction"] in ("UP", "DOWN")
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "explanation" in data
        assert "disclaimer" in data

    def test_predict_model_selection_lightgbm(self, app_client):
        """Verify model_name field actually switches which model is used."""
        feature_row = self._mock_feature_row()

        with patch(
            "market_intelligence.api.predict._get_feature_row",
            return_value=(feature_row, True),
        ):
            data = app_client.post(
                "/api/v1/predict",
                json={"symbol": "AAPL", "model_name": "lightgbm"},
            ).json()
        assert data.get("model_name") == "lightgbm"

    def test_predict_saves_to_db(self, app_client):
        """Verify prediction is persisted to PostgreSQL (FR-DB-001)."""
        feature_row = self._mock_feature_row()

        with patch(
            "market_intelligence.api.predict._get_feature_row",
            return_value=(feature_row, True),
        ):
            app_client.post(
                "/api/v1/predict",
                json={"symbol": "SYTEST", "model_name": "xgboost"},
            )

        from market_intelligence.db.session import engine

        result = pd.read_sql(
            "SELECT * FROM predictions WHERE symbol = 'SYTEST' LIMIT 1", engine
        )
        assert len(result) == 1
        assert result["model_name"].iloc[0] == "xgboost"

    def test_predict_logs_api_call(self, app_client):
        """Verify API call is logged (FR-DB-004)."""
        feature_row = self._mock_feature_row()

        with patch(
            "market_intelligence.api.predict._get_feature_row",
            return_value=(feature_row, True),
        ):
            app_client.post(
                "/api/v1/predict",
                json={"symbol": "LOGTEST", "model_name": "xgboost"},
            )

        from market_intelligence.db.session import engine

        logs = pd.read_sql(
            "SELECT * FROM api_logs WHERE endpoint = '/api/v1/predict' ORDER BY id DESC LIMIT 1",
            engine,
        )
        assert len(logs) >= 1
