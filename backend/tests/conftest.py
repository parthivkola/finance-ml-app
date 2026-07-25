# tests/conftest.py
"""Shared pytest fixtures for unit, integration and system tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def sample_prices() -> pd.DataFrame:
    """Minimal OHLCV DataFrame to exercise feature building without API calls."""
    n = 120  # need > 50 for SMA_50
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    import numpy as np

    rng = np.random.default_rng(42)
    close = 200 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - rng.uniform(0, 2, n),
            "high": close + rng.uniform(0, 3, n),
            "low": close - rng.uniform(0, 3, n),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(int),
        }
    )


@pytest.fixture(scope="session")
def app_client():
    """TestClient wrapping the FastAPI app — no live server needed."""
    from market_intelligence.api.main import app

    return TestClient(app)
