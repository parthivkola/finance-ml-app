import pandas as pd

def align_and_merge(prices_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges prices and news. Handles weekend news by pushing it to Monday.
    """
    # 1. Align weekend news to the next Monday
    # dayofweek: Monday=0, Sunday=6
    news_df['aligned_date'] = news_df['date'].apply(
        lambda d: d + pd.Timedelta(days=2) if d.weekday() == 5 else (
                  d + pd.Timedelta(days=1) if d.weekday() == 6 else d)
    )
    
    # 2. Convert to datetime so we can merge
    prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
    news_df['aligned_date'] = pd.to_datetime(news_df['aligned_date']).dt.date
    
    # 3. Merge!
    merged_df = pd.merge(
        prices_df, 
        news_df, 
        left_on='date', 
        right_on='aligned_date', 
        how='left'
    )
    
    # Forward fill missing prices for days where we have news but no trades (e.g. holidays)
    merged_df.ffill(inplace=True)
    
    return merged_df