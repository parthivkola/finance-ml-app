"""
Scheduler — hourly price refresh + auto-retrain + 15-minute live news via RSS.
"""
from __future__ import annotations

import logging

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

WATCH_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
PRICE_HISTORY_DAYS = 3650  # 10 years — gives ~2500 rows per symbol

# Retrain when this many NEW price rows have arrived since last training
RETRAIN_THRESHOLD = 30


# ─── News (runs every 15 minutes via RSS — free, live, no API key) ────────────

def refresh_news_all_symbols() -> None:
    """
    Fetch fresh RSS news for all watched symbols and score any unscored articles.
    Runs every 15 minutes for near-live updates.
    """
    from market_intelligence.data.rss_news import fetch_rss_news
    from market_intelligence.data.storage import save_news
    from market_intelligence.nlp.process_news import process_unscored_news

    logger.info("=== 15-min news refresh started ===")
    for symbol in WATCH_SYMBOLS:
        try:
            df = fetch_rss_news(symbol)
            if not df.empty:
                save_news(df, symbol)
                logger.info("RSS: saved %d articles for %s", len(df), symbol)
        except Exception as exc:
            logger.error("RSS news failed for %s: %s", symbol, exc)

    try:
        process_unscored_news()
    except Exception as exc:
        logger.error("News scoring failed: %s", exc)

    logger.info("=== 15-min news refresh complete ===")


# ─── Prices (runs every hour) ─────────────────────────────────────────────────

def _refresh_prices(symbol: str) -> None:
    from market_intelligence.data.yahoo_finance import fetch_prices
    from market_intelligence.data.storage import save_prices

    try:
        prices = fetch_prices(symbol, days=PRICE_HISTORY_DAYS)
        save_prices(prices, symbol)
    except Exception as exc:
        logger.error("Price fetch failed for %s: %s", symbol, exc)


def _should_retrain() -> bool:
    from market_intelligence.db.session import engine

    try:
        total = pd.read_sql("SELECT COUNT(*) AS n FROM stock_prices", engine)["n"].iloc[0]
        last = pd.read_sql(
            "SELECT total_price_rows FROM model_registry "
            "WHERE total_price_rows IS NOT NULL ORDER BY trained_at DESC LIMIT 1",
            engine,
        )
        if last.empty:
            return True
        return int(total) - int(last["total_price_rows"].iloc[0]) >= RETRAIN_THRESHOLD
    except Exception as exc:
        logger.error("Retrain check failed: %s", exc)
        return False


def _retrain() -> None:
    """Load all data from DB and retrain all models."""
    from market_intelligence.db.session import engine
    from market_intelligence.ml.trainer import train_all

    logger.info("🔁 Auto-retrain triggered …")
    try:
        prices_df = pd.read_sql("SELECT * FROM stock_prices ORDER BY symbol, date", engine)
        news_df = pd.read_sql(
            """
            SELECT symbol,
                   DATE(published_date) AS date,
                   AVG(sentiment_score)  AS sentiment_score
            FROM   news_articles
            WHERE  sentiment_score IS NOT NULL
            GROUP  BY symbol, DATE(published_date)
            """,
            engine,
        )
        news_df["date"] = pd.to_datetime(news_df["date"])
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        # Merge per symbol+date — each stock gets its own news sentiment
        merged = prices_df.merge(news_df, on=["symbol", "date"], how="left")

        if len(merged) < 100:
            logger.warning("Not enough data to retrain (%d rows). Skipping.", len(merged))
            return

        results = train_all(merged)
        logger.info("✅ Retrain complete: %s", list(results.keys()))
    except Exception as exc:
        logger.error("Auto-retrain failed: %s", exc)


def refresh_prices_and_retrain() -> None:
    logger.info("=== Hourly price refresh started ===")
    for symbol in WATCH_SYMBOLS:
        _refresh_prices(symbol)

    if _should_retrain():
        _retrain()

    logger.info("=== Hourly price refresh complete ===")


# ─── Daily cleanup (disk + DB) ────────────────────────────────────────────────

def daily_cleanup() -> None:
    """
    Runs daily at 02:00 UTC. Purges:
    - Old API logs and prediction records older than 90 days (keeps DB lean)
    - Docker build cache (keeps disk space free on t3.micro)
    - Python garbage collection pass
    """
    import gc
    import subprocess
    from market_intelligence.db.session import engine
    from sqlalchemy import text

    logger.info("=== Daily cleanup started ===")

    # 1. Prune old DB records
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "DELETE FROM api_logs WHERE requested_at < NOW() - INTERVAL '90 days'"
            ))
            conn.execute(text(
                "DELETE FROM predictions WHERE predicted_at < NOW() - INTERVAL '90 days'"
            ))
            conn.commit()
            logger.info("DB cleanup: removed old logs and predictions")
    except Exception as exc:
        logger.error("DB cleanup failed: %s", exc)

    # 2. Prune Docker build cache (free disk space)
    try:
        result = subprocess.run(
            ["docker", "builder", "prune", "-f", "--keep-storage", "500mb"],
            capture_output=True, text=True, timeout=60
        )
        logger.info("Docker prune: %s", result.stdout.strip() or "done")
    except Exception as exc:
        logger.warning("Docker prune skipped (not available in container): %s", exc)

    # 3. Python GC pass
    collected = gc.collect()
    logger.info("GC: collected %d objects", collected)

    logger.info("=== Daily cleanup complete ===")


# ─── Scheduler setup ──────────────────────────────────────────────────────────

def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")

    # News: every 15 minutes → live updates
    scheduler.add_job(
        refresh_news_all_symbols,
        trigger=IntervalTrigger(minutes=15),
        id="news_refresh",
        name="15-min RSS News Refresh",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=120,
    )

    # Prices + retrain: every hour
    scheduler.add_job(
        refresh_prices_and_retrain,
        trigger=IntervalTrigger(hours=1),
        id="price_refresh",
        name="Hourly Price Refresh + Auto-Retrain",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )

    # Daily cleanup: DB records + Docker build cache at 02:00 UTC
    from apscheduler.triggers.cron import CronTrigger
    scheduler.add_job(
        daily_cleanup,
        trigger=CronTrigger(hour=2, minute=0, timezone="UTC"),
        id="daily_cleanup",
        name="Daily DB + Disk Cleanup",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=600,
    )

    return scheduler
