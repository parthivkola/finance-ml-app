"""
Additional API routers — satisfies FR-API-002, FR-API-003, FR-API-005.

GET /api/v1/news/{symbol}       — recent scored news articles
GET /api/v1/models              — all trained models with metrics (model selection support)
GET /api/v1/history/{symbol}    — historical price records
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from market_intelligence.api.schemas import HistoryResponse, ModelMetrics, NewsResponse
from market_intelligence.db.models import ModelRegistry, NewsArticle, StockPrice
from market_intelligence.db.session import SessionLocal

router = APIRouter()


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
