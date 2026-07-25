import pandas as pd

def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts FinBERT string labels to numerical values and takes the mean per day.
    """
    # Map strings to ML-friendly integers
    label_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    
    news_df['sentiment_value'] = news_df['sentiment_label'].map(label_map)
    
    # Weight the sentiment by the model's confidence score
    news_df['weighted_sentiment'] = news_df['sentiment_value'] * news_df['sentiment_score']
    
    # Group by date and take the average sentiment for that day
    daily_sentiment = news_df.groupby('aligned_date')['weighted_sentiment'].mean().reset_index()
    
    return daily_sentiment
