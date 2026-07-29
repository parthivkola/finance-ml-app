"""
Additional API routers — satisfies FR-API-002, FR-API-003, FR-API-005.

GET /api/v1/news/{symbol}       — recent scored news articles
GET /api/v1/models              — all trained models with metrics (model selection support)
GET /api/v1/models/best         — best non-overfit model by accuracy (dynamic champion)
GET /api/v1/history/{symbol}    — historical price records
"""
from __future__ import annotations
import re

from fastapi import APIRouter, HTTPException

from market_intelligence.api.schemas import BestModelResponse, HistoryResponse, ModelMetrics, NewsResponse
from market_intelligence.db.models import ModelRegistry, NewsArticle, StockPrice
from market_intelligence.db.session import SessionLocal

router = APIRouter()

# Valid model name pattern — must match trainer output
_VALID_MODEL_RE = re.compile(r'^(xgboost|lightgbm|random_forest|logistic_regression)_(1|3|5)d$')


@router.get("/news/{symbol}", response_model=list[NewsResponse], tags=["News"])
def get_news(symbol: str, limit: int = 20):
    """Return the most recent scored news articles for a symbol (FR-API-002)."""
    db = SessionLocal()
    try:
        symbol = symbol.upper().strip().replace(".", "-")
        articles = (
            db.query(NewsArticle)
            .filter(NewsArticle.symbol == symbol)
            .order_by(NewsArticle.published_date.desc())
            .limit(limit)
            .all()
        )
        return [
            NewsResponse(
                id=a.id,
                symbol=a.symbol,
                title=a.title,
                url=a.url,
                published_date=str(a.published_date),
                sentiment_score=a.sentiment_score,
                sentiment_label=a.sentiment_label,
            )
            for a in articles
        ]
    finally:
        db.close()


@router.get("/models/best", response_model=BestModelResponse, tags=["Models"])
def get_best_model():
    """
    Dynamically elect the champion model:
    1. Only consider models that have a valid name (e.g. xgboost_1d)
    2. Exclude any model whose overfit_status contains 'OVERFIT'
    3. Among the remainder, pick the one with the highest test accuracy
    4. Falls back to highest accuracy overall if all are flagged overfit
    """
    from sqlalchemy import func
    db = SessionLocal()
    try:
        subq = (
            db.query(
                ModelRegistry.model_name,
                func.max(ModelRegistry.trained_at).label("max_at"),
            )
            .group_by(ModelRegistry.model_name)
            .subquery()
        )
        records = (
            db.query(ModelRegistry)
            .join(
                subq,
                (ModelRegistry.model_name == subq.c.model_name)
                & (ModelRegistry.trained_at == subq.c.max_at),
            )
            .all()
        )

        # Filter to valid multi-horizon model names only
        valid = [r for r in records if _VALID_MODEL_RE.match(r.model_name or "")]
        if not valid:
            raise HTTPException(status_code=404, detail="No trained models found. Run training first.")

        # Prefer non-overfit models; fall back to all if every model is flagged
        non_overfit = [r for r in valid if "OVERFIT" not in (r.overfit_status or "")]
        candidates = non_overfit if non_overfit else valid

        # Champion = highest test accuracy; tie-break by ROC-AUC then F1
        champion = max(
            candidates,
            key=lambda r: (
                r.accuracy or 0.0,
                r.roc_auc or 0.0,
                r.f1_score or 0.0,
            ),
        )

        return BestModelResponse(
            model_name=champion.model_name,
            version=champion.version,
            accuracy=champion.accuracy,
            train_accuracy=champion.train_accuracy,
            overfit_status=champion.overfit_status,
            f1_score=champion.f1_score,
            roc_auc=champion.roc_auc,
            is_fallback=len(non_overfit) == 0,
        )
    finally:
        db.close()


@router.get("/models", response_model=list[ModelMetrics], tags=["Models"])
def get_models():
    """
    Return the LATEST trained entry per model with accuracy, F1, and ROC-AUC (FR-API-003).
    """
    from sqlalchemy import func
    db = SessionLocal()
    try:
        # Subquery: latest trained_at per model_name
        subq = (
            db.query(
                ModelRegistry.model_name,
                func.max(ModelRegistry.trained_at).label("max_at"),
            )
            .group_by(ModelRegistry.model_name)
            .subquery()
        )
        records = (
            db.query(ModelRegistry)
            .join(
                subq,
                (ModelRegistry.model_name == subq.c.model_name)
                & (ModelRegistry.trained_at == subq.c.max_at),
            )
            .order_by(ModelRegistry.model_name)
            .all()
        )
        return [
            ModelMetrics(
                id=r.id,
                model_name=r.model_name,
                version=r.version,
                trained_at=str(r.trained_at),
                accuracy=r.accuracy,
                train_accuracy=r.train_accuracy,
                overfit_status=r.overfit_status,
                f1_score=r.f1_score,
                roc_auc=r.roc_auc,
                artifact_path=r.artifact_path,
            )
            for r in records
        ]
    finally:
        db.close()



@router.get("/history/{symbol}", response_model=list[HistoryResponse], tags=["Prices"])
def get_history(symbol: str, limit: int = 90):
    """Return historical OHLCV records for a symbol (FR-API-005)."""
    db = SessionLocal()
    try:
        symbol = symbol.upper().strip().replace(".", "-")
        records = (
            db.query(StockPrice)
            .filter(StockPrice.symbol == symbol)
            .order_by(StockPrice.date.desc())
            .limit(limit)
            .all()
        )
        if not records:
            from market_intelligence.data.yahoo_finance import fetch_prices
            from market_intelligence.data.storage import save_prices
            try:
                prices_df = fetch_prices(symbol, days=limit+30)
                if not prices_df.empty:
                    prices_df.columns = [c.lower() for c in prices_df.columns]
                    save_prices(prices_df, symbol)
                    
                    # Fetch again
                    records = (
                        db.query(StockPrice)
                        .filter(StockPrice.symbol == symbol)
                        .order_by(StockPrice.date.desc())
                        .limit(limit)
                        .all()
                    )
            except Exception:
                pass
                
            if not records:
                raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

        return [
            HistoryResponse(
                symbol=r.symbol,
                date=str(r.date),
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
            )
            for r in records
        ]
    finally:
        db.close()
