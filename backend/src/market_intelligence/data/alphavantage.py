import os
import hashlib
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Load .env from project root
load_dotenv(PROJECT_ROOT.parent / ".env")
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


def fetch_news(symbol: str) -> pd.DataFrame:
    """
    Fetches news sentiment data from AlphaVantage and deduplicates articles.
    Saves a Parquet snapshot to data/raw/ for offline reuse.
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={symbol}&apikey={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    if "feed" not in data:
        print(f"No news returned for {symbol}. Response: {list(data.keys())}")
        return pd.DataFrame()

    articles = []
    seen_hashes = set()

    for item in data["feed"]:
        url_hash = hashlib.sha256(item["url"].encode()).hexdigest()

        # Skip duplicates
        if url_hash in seen_hashes:
            continue
        seen_hashes.add(url_hash)

        # Parse date from AlphaVantage format: '20231024T133000'
        pub_date = pd.to_datetime(item["time_published"], format="%Y%m%dT%H%M%S")

        articles.append({
            "url_hash": url_hash,
            "title": item["title"],
            "published_date": pub_date,
            "date": pub_date.date(),  # raw date used for merging
            "summary": item["summary"],
        })

    df = pd.DataFrame(articles)

    # Persist raw data to disk
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{symbol}_news.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}")

    return df
