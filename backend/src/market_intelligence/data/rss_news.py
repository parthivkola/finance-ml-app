"""
RSS News Fetcher — free, unlimited, live news updates every 15 minutes.

Sources used (no API key required):
  - Google News RSS (financial topics per symbol)
  - Yahoo Finance RSS

Falls back to AlphaVantage only if RSS yields nothing.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import feedparser
import pandas as pd

logger = logging.getLogger(__name__)

# RSS feeds per symbol — Google News financial RSS is reliable and free
RSS_FEEDS = {
    "__template__": [
        "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
    ]
}


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:64]


def _parse_date(entry) -> datetime:
    """Safely parse published date from RSS entry."""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


def fetch_rss_news(symbol: str) -> pd.DataFrame:
    """
    Fetch recent news articles for a symbol from free RSS feeds.
    Returns a DataFrame compatible with storage.save_news().
    """
    symbol_upper = symbol.upper()
    articles = []

    feeds = [url.format(symbol=symbol_upper) for url in RSS_FEEDS["__template__"]]

    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
            if parsed.bozo and not parsed.entries:
                logger.warning("RSS parse error for %s: %s", feed_url, parsed.bozo_exception)
                continue

            for entry in parsed.entries[:20]:  # Max 20 articles per feed
                title = getattr(entry, "title", "").strip()
                link = getattr(entry, "link", feed_url)
                summary = getattr(entry, "summary", "").strip()

                if not title:
                    continue
                    
                # Strict relevance check
                company_map = {
                    "AAPL": "apple", "MSFT": "microsoft", "GOOGL": "google", 
                    "AMZN": "amazon", "TSLA": "tesla", "NVDA": "nvidia", "META": "meta"
                }
                c_name = company_map.get(symbol_upper, symbol_upper.lower())
                t_lower = title.lower()
                s_lower = summary.lower() if summary else ""
                
                # Check if symbol or company name is in the title/summary
                if symbol_upper not in title and symbol_upper not in (summary or ""):
                    if c_name not in t_lower and c_name not in s_lower:
                        continue

                articles.append({
                    "url_hash": _url_hash(link),
                    "url": link,
                    "title": title[:500],
                    "summary": summary[:1000] if summary else None,
                    "published_date": _parse_date(entry),
                    "source": "RSS",
                })

            logger.info("RSS: fetched %d articles for %s from %s",
                        len(parsed.entries), symbol_upper, feed_url)

        except Exception as exc:
            logger.error("RSS fetch failed for %s (%s): %s", symbol_upper, feed_url, exc)

    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset=["url_hash"])
    return df
