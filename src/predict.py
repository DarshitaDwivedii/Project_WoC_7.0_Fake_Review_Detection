"""Loads the best trained model + TF-IDF vectorizer and exposes predict()."""

import os
import pickle

from preprocessing.classical import preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    VECTORIZER = pickle.load(f)

with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)

with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
    MODEL_NAME = f.read().strip()


def predict(text: str) -> dict:
    clean = preprocess(text)
    vec = VECTORIZER.transform([clean])

    pred = int(MODEL.predict(vec)[0])  # 1 = fake (CG), 0 = real (OR)
    confidence = None
    if hasattr(MODEL, "predict_proba"):
        confidence = float(MODEL.predict_proba(vec)[0][pred])
    elif hasattr(MODEL, "decision_function"):
        import math
        score = float(MODEL.decision_function(vec)[0])
        confidence = 1 / (1 + math.exp(-abs(score)))  # sigmoid-squash the margin

    return {
        "label": "fake" if pred == 1 else "real",
        "is_fake": bool(pred),
        "confidence": round(confidence, 4) if confidence is not None else None,
        "model_used": MODEL_NAME,
    }


if __name__ == "__main__":
    print(predict("This product changed my life!! Best purchase ever!!! 5 stars!!!"))
    print(predict("Decent quality for the price, arrived a day late but works as expected."))
