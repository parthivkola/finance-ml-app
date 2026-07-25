import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from market_intelligence.db.session import SessionLocal
from market_intelligence.db.models import StockPrice, NewsArticle


def save_prices(df: pd.DataFrame, symbol: str) -> None:
    """
    Upserts price rows into the stock_prices table.
    Uses ON CONFLICT DO NOTHING based on (symbol, date) — safe to re-run.
    """
    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            stmt = (
                insert(StockPrice)
                .values(
                    symbol=symbol,
                    date=row["date"],
                    open=float(row.get("Open", row.get("open", 0))),
                    high=float(row.get("High", row.get("high", 0))),
                    low=float(row.get("Low", row.get("low", 0))),
                    close=float(row.get("Close", row.get("close", 0))),
                    volume=int(row.get("Volume", row.get("volume", 0))),
                )
                .on_conflict_do_nothing(index_elements=["symbol", "date"])
            )
            db.execute(stmt)
        db.commit()
        print(f"Saved {len(df)} price rows for {symbol} → PostgreSQL")
    finally:
        db.close()


def save_news(df: pd.DataFrame, symbol: str) -> None:
    """
    Upserts news articles into the news_articles table.
    url_hash is a unique constraint — duplicates are silently skipped.
    """
    if df.empty:
        return

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            stmt = (
                insert(NewsArticle)
                .values(
                    symbol=symbol,
                    url_hash=row["url_hash"],
                    url=row.get("url"),
                    title=row["title"],
                    published_date=row["published_date"],
                    sentiment_score=None,  # filled in Phase 4 after NLP
                )
                .on_conflict_do_nothing(index_elements=["url_hash"])
            )
            db.execute(stmt)
        db.commit()
        print(f"Saved {len(df)} news rows for {symbol} → PostgreSQL")
    finally:
        db.close()
