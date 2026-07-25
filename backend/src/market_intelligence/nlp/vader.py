import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download lexicon on first run
nltk.download('vader_lexicon', quiet=True)

class VaderAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)
        
        # VADER returns a 'compound' score from -1 (very neg) to +1 (very pos)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
            
        return {
            "label": label,
            "score": abs(compound),
            "raw_scores": scores
        }
