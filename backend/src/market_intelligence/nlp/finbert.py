import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FinBERTAnalyzer:
    def __init__(self):
        # We load the weights from HuggingFace
        model_name = "ProsusAI/finbert"
        
        # Tokenizer converts words into numerical vectors
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # The actual neural network
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Label mapping specific to this model
        self.labels = ['positive', 'negative', 'neutral']

    def analyze(self, text: str) -> dict:
        """
        Takes an article summary and returns sentiment scores.
        """
        # 1. Tokenize the input text. Truncate if it's too long for BERT (max 512 tokens)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # 2. Run inference without calculating gradients (saves massive amounts of memory)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 3. Apply softmax to convert raw logits into percentages (0.0 to 1.0)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 4. Get the highest probability
        winner_idx = torch.argmax(probabilities).item()
        
        return {
            "label": self.labels[winner_idx],
            "score": probabilities[0][winner_idx].item(), # Confidence score
            "raw_scores": probabilities[0].tolist()
        }
