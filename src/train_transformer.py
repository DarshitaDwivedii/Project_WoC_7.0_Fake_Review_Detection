"""
Fine-tunes DistilBERT for fake review classification.

NOT RUN IN THE BUILD SANDBOX — no GPU was available there. Written to be
CPU-runnable on a laptop by default (small subsample, short sequences, few
epochs). Expected time on a modern laptop CPU: ~20-40 min for the defaults
below. If you have a GPU (or want to use the full 40k dataset), bump
TRAIN_SAMPLE_SIZE / EPOCHS, or run this in Google Colab with a free GPU
runtime instead — the code doesn't change either way, torch will just pick
up the GPU automatically if `torch.cuda.is_available()`.

Install first (not in the base requirements.txt — heavy, optional):
    pip install torch transformers datasets

Run:
    python src/train_transformer.py
"""

import os

import evaluate
import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from preprocessing.transformer_lite import preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "fakeReviewData.csv")
MODEL_OUT_DIR = os.path.join(ROOT, "models", "distilbert_fake_review")

MODEL_NAME = "distilbert-base-uncased"

# --- Tune these based on your hardware ---
TRAIN_SAMPLE_SIZE = 6000   # None = use full ~32k train split (needs a GPU realistically)
MAX_LENGTH = 128
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
# ------------------------------------------

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")
mlflow.set_experiment("fake-review-detection-transformer")


def load_data():
    df = pd.read_csv(DATA_PATH).dropna(subset=["text_", "label"])
    df["text"] = df["text_"].apply(preprocess)
    df["target"] = (df["label"] == "CG").astype(int)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )
    if TRAIN_SAMPLE_SIZE is not None and TRAIN_SAMPLE_SIZE < len(train_df):
        train_df = train_df.groupby("target", group_keys=False).apply(
            lambda g: g.sample(TRAIN_SAMPLE_SIZE // 2, random_state=42)
        )
    return train_df, test_df


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_df, test_df = load_data()
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", "target"]].rename(columns={"target": "label"}))
    test_ds = Dataset.from_pandas(test_df[["text", "target"]].rename(columns={"target": "label"}))
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "precision": precision.compute(predictions=preds, references=labels)["precision"],
            "recall": recall.compute(predictions=preds, references=labels)["recall"],
            "f1": f1.compute(predictions=preds, references=labels)["f1"],
        }

    args = TrainingArguments(
        output_dir=os.path.join(ROOT, "checkpoints_tmp"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name="distilbert"):
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "train_sample_size": len(train_df),
                "max_length": MAX_LENGTH,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "device": device,
            }
        )

        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        print("\nFinal metrics:", metrics)

        os.makedirs(MODEL_OUT_DIR, exist_ok=True)
        trainer.save_model(MODEL_OUT_DIR)
        tokenizer.save_pretrained(MODEL_OUT_DIR)
        print(f"\nModel saved to {MODEL_OUT_DIR}")


if __name__ == "__main__":
    main()
