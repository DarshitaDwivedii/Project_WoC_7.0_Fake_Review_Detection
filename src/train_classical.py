"""
Trains 5 classical models on TF-IDF features, logs hyperparameters + metrics
for every run to MLflow, and saves the best model + vectorizer to disk.

Run:
    python src/train_classical.py
Then view results:
    mlflow ui --backend-store-uri ./mlruns
"""

import json
import os
import pickle
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from preprocessing.classical import preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "fakeReviewData.csv")
MODELS_DIR = os.path.join(ROOT, "models")
MLRUNS_DIR = os.path.join(ROOT, "mlruns")

os.makedirs(MODELS_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")
mlflow.set_experiment("fake-review-detection-classical")

MODEL_REGISTRY = {
    "logistic_regression": {
        "estimator": LogisticRegression,
        "params": {"C": 1.0, "max_iter": 1000, "solver": "liblinear"},
    },
    "linear_svm": {
        "estimator": LinearSVC,
        "params": {"C": 1.0, "max_iter": 2000},
    },
    "random_forest": {
        "estimator": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 25, "n_jobs": 2, "random_state": 42},
    },
    "naive_bayes": {
        "estimator": MultinomialNB,
        "params": {"alpha": 0.5},
    },
    "xgboost": {
        "estimator": XGBClassifier,
        "params": {
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": 42,
        },
    },
}

TFIDF_PARAMS = {"max_features": 5000, "ngram_range": (1, 2), "min_df": 3}


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text_", "label"])
    df["clean_text"] = df["text_"].apply(preprocess)
    df["target"] = (df["label"] == "CG").astype(int)  # 1 = computer-generated (fake)
    return df


def main():
    print("Loading + preprocessing data...")
    df = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    results = {}
    best_name, best_f1, best_model = None, -1, None

    for name, cfg in MODEL_REGISTRY.items():
        print(f"\nTraining {name}...")
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_params(cfg["params"])
            mlflow.log_params({f"tfidf_{k}": v for k, v in TFIDF_PARAMS.items()})

            t0 = time.time()
            model = cfg["estimator"](**cfg["params"])
            model.fit(X_train_vec, y_train)
            train_time = time.time() - t0

            preds = model.predict(X_test_vec)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            cm = confusion_matrix(y_test, preds).tolist()

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("train_time_sec", train_time)
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                print(f"  (skipping mlflow model artifact logging: {e})")

            results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "train_time_sec": train_time,
                "hyperparameters": cfg["params"],
            }
            print(f"  acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} time={train_time:.1f}s")

            if f1 > best_f1:
                best_f1, best_name, best_model = f1, name, model

    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    print(f"\nBest model: {best_name} (f1={best_f1:.4f}) -> saved to models/best_model.pkl")
    print("Full comparison saved to models/results.json")


if __name__ == "__main__":
    main()
