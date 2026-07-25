"""Pydantic schemas — satisfies FR-API-006. Supports model selection for frontend."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────── Request bodies ──────────────────────────────────

class PredictRequest(BaseModel):
    symbol: str = Field(..., example="AAPL")
    model_name: Optional[str] = Field(
        default="xgboost",
        description="Model to use: 'xgboost', 'lightgbm', 'random_forest', 'logistic_regression'",
        example="xgboost",
    )


# ──────────────────────────── Response bodies ─────────────────────────────────

class FeatureImpact(BaseModel):
    feature: str
    impact: float


class TechnicalIndicators(BaseModel):
    sma_20: Optional[float]
    sma_50: Optional[float]
    rsi: Optional[float]
    macd: Optional[float]


class PredictResponse(BaseModel):
    symbol: str
    model_name: str
    prediction: str           # "UP" or "DOWN"
    confidence: float         # 0.0 – 1.0
    disclaimer: str = "For research and educational purposes only. Not financial advice."
    explanation: list[FeatureImpact]
    indicators: Optional[TechnicalIndicators] = None


class ModelMetrics(BaseModel):
    id: int
    model_name: str
    version: str
    trained_at: str
    accuracy: Optional[float]
    f1_score: Optional[float]
    roc_auc: Optional[float]
    artifact_path: str


class NewsResponse(BaseModel):
    id: int
    symbol: str
    title: str
    url: Optional[str] = None
    published_date: str
    sentiment_score: Optional[float]
    sentiment_label: Optional[str]


class HistoryResponse(BaseModel):
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    models_available: list[str]
