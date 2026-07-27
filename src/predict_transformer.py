"""Loads the fine-tuned DistilBERT model (run train_transformer.py first)
and exposes predict(). Mirrors predict.py's interface so api/main.py could
swap to this instead.

NOT RUN IN THE BUILD SANDBOX — no GPU available there. Test locally after
running train_transformer.py.
"""

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing.transformer_lite import preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "distilbert_fake_review")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
        _model.eval()


def predict(text: str) -> dict:
    _load()
    clean = preprocess(text)
    inputs = _tokenizer(clean, return_tensors="pt", truncation=True, max_length=128).to(_device)

    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred = int(torch.argmax(probs).item())
    confidence = float(probs[pred].item())

    return {
        "label": "fake" if pred == 1 else "real",
        "is_fake": bool(pred),
        "confidence": round(confidence, 4),
        "model_used": "distilbert",
    }


if __name__ == "__main__":
    print(predict("This product changed my life!! Best purchase ever!!! 5 stars!!!"))
    print(predict("Decent quality for the price, arrived a day late but works as expected."))
