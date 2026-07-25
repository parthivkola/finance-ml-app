import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


# Root of the project (5 levels up from this file)
# yahoo_finance.py → data → market_intelligence → src → backend → project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def fetch_prices(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Downloads OHLCV data for a given symbol over the last X days.
    Saves a Parquet snapshot to data/raw/ for offline reuse.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    print(f"Fetching prices for {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)

    # yfinance returns multi-index columns in newer versions — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date"}, inplace=True)

    # Persist raw data to disk so we don't burn API calls on every run
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{symbol}_prices.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}")

    return df
