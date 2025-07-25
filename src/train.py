#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import inspect

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class AbstractDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds).tolist()
    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}


def main(args):
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")

    with open(data_dir / "label_map.json", "r") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = {k: int(v) for k, v in label_map["label2id"].items()}

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = AbstractDataset(train_df, tokenizer, max_len=args.max_len)
    val_dataset = AbstractDataset(val_df, tokenizer, max_len=args.max_len)

    # ---- Build TrainingArguments kwargs dynamically based on your installed version ----
    sig = inspect.signature(TrainingArguments.__init__)
    valid = sig.parameters.keys()

    base_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "save_total_limit": 2,
    }

    # Newer versions support these:
    if "evaluation_strategy" in valid:
        base_kwargs.update(
            {
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch" if "save_strategy" in valid else "epoch",
                "load_best_model_at_end": True
                if "load_best_model_at_end" in valid
                else False,
                "metric_for_best_model": "f1"
                if "metric_for_best_model" in valid
                else None,
            }
        )

    # Remove Nones & invalid keys
    clean_kwargs = {k: v for k, v in base_kwargs.items() if v is not None and k in valid}

    training_args = TrainingArguments(**clean_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    metrics_path = Path(args.output_dir) / "baseline_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n[Done] Baseline model saved to: {args.output_dir}")
    print(f"Metrics:\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline DistilBERT model")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="models/baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()
    main(args)
