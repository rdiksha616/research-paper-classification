#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class AbstractDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx]),
        }


def plot_confusion_matrix(cm, labels, output_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(args):
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    test_path = data_dir / "test.csv"
    label_map_path = data_dir / "label_map.json"

    # Load dataset
    df = pd.read_csv(test_path)
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = {k: int(v) for k, v in label_map["label2id"].items()}

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Prepare dataset
    dataset = AbstractDataset(df, tokenizer, max_len=args.max_len)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for item in dataset:
            input_ids = item["input_ids"].unsqueeze(0)
            att_mask = item["attention_mask"].unsqueeze(0)
            labels = item["labels"].item()

            outputs = model(input_ids, attention_mask=att_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()

            all_preds.append(pred)
            all_labels.append(labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(all_labels, all_preds, target_names=list(label2id.keys())),
    }

    metrics_path = model_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cm_path = model_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, list(label2id.keys()), cm_path)

    print(f"[Done] Test metrics saved to: {metrics_path}")
    print(f"[Done] Confusion matrix saved to: {cm_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_dir", type=str, default="models/baseline")
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()
    main(args)
