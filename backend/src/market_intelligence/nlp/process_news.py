import sys
from pathlib import Path
from sqlalchemy.orm import Session

# Add src to path if running directly
src_path = Path(__file__).resolve().parents[2]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from market_intelligence.db.session import SessionLocal  # noqa: E402
from market_intelligence.db.models import NewsArticle  # noqa: E402
from market_intelligence.nlp.vader import VaderAnalyzer  # noqa: E402

try:
    from market_intelligence.nlp.finbert import FinBERTAnalyzer  # noqa: E402
    HAS_FINBERT = True
except Exception as e:
    print(f"Warning: Could not import FinBERTAnalyzer ({e}). Will use VADER exclusively.")
    HAS_FINBERT = False

def process_unscored_news():
    """
    Finds all news articles in the DB without a sentiment score,
    runs them through FinBERT, and updates the database.
    If FinBERT fails (e.g., out of memory or HuggingFace error), it falls back to VADER.
    """
    db: Session = SessionLocal()
    
    analyzer = None
    if HAS_FINBERT:
        try:
            analyzer = FinBERTAnalyzer()
        except Exception as e:
            print(f"Warning: Failed to initialize FinBERT ({e}). Falling back to VADER.")
            
    vader_analyzer = VaderAnalyzer()
    
    try:
        # Get all articles where sentiment_score is NULL
        unscored_articles = db.query(NewsArticle).filter(NewsArticle.sentiment_score.is_(None)).all()
        
        if not unscored_articles:
            print("No new articles to score! Everything is up to date.")
            return

        print(f"Found {len(unscored_articles)} unscored articles. Starting Sentiment Analysis...")
        
        for idx, article in enumerate(unscored_articles, 1):
            # We use the title for context (summary is not saved in DB)
            text_to_analyze = article.title
            
            used_model = "FinBERT"
            try:
                if analyzer is None:
                    raise RuntimeError("FinBERT is not available.")
                # 1. Try heavy Deep Learning model first
                result = analyzer.analyze(text_to_analyze)
            except Exception as e:
                # 2. Fallback to lightning-fast rule-based system if FinBERT fails
                print(f"\\n[Warning] FinBERT failed on article {article.id}: {e}. Falling back to VADER.")
                result = vader_analyzer.analyze(text_to_analyze)
                used_model = "VADER"
            
            # Map sentiment to a numerical score for ML (-1.0 to 1.0)
            label = result['label']  # 'positive', 'negative', 'neutral'
            direction = 1 if label == 'positive' else (-1 if label == 'negative' else 0)
            final_score = result['score'] * direction

            # Update the database record — score + label + which model scored it
            article.sentiment_score = final_score
            article.sentiment_label = label
            article.sentiment_model = used_model

            print(f"[{idx}/{len(unscored_articles)}] Scored: {final_score:+.2f} | {label:8} | {used_model} | {article.title[:50]}...")
            
        # Commit all changes to the database
        db.commit()
        print("\nSuccessfully updated all articles in the database!")
        
    finally:
        db.close()

if __name__ == "__main__":
    process_unscored_news()
