"""
DistilBERT sentiment analyzer — satisfies FR-NLP-002.
Uses 'distilbert-base-uncased-finetuned-sst-2-english', a lighter
sentiment model than FinBERT (67M params vs 110M) for faster inference.
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DistilBERTAnalyzer:
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        # SST-2 labels: 0 = negative, 1 = positive
        self.labels = ["negative", "positive"]

    def analyze(self, text: str) -> dict:
        """
        Returns sentiment label, confidence score, and raw logit scores.
        Maps binary SST-2 output to positive / negative / neutral using
        a confidence threshold (<60% → neutral).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        winner_idx = torch.argmax(probabilities).item()
        confidence = probabilities[0][winner_idx].item()

        # Map to 3-class with confidence threshold
        if confidence < 0.60:
            label = "neutral"
        else:
            label = self.labels[winner_idx]

        return {
            "label": label,
            "score": confidence,
            "raw_scores": probabilities[0].tolist(),
        }
