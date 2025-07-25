#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# PEFT / LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------------
# Dataset
# -----------------------------
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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# -----------------------------
# Utilities
# -----------------------------
def select_default_target_modules(model_name: str) -> List[str]:
    """Return a decent default list of target modules for common encoder families."""
    name = model_name.lower()
    if "distilbert" in name:
        # Works for DistilBERT
        return ["q_lin", "v_lin"]
    if "bert" in name or "roberta" in name or "electra" in name:
        # Common for BERT-ish models
        return ["query", "key", "value", "dense"]
    if "llama" in name or "mistral" in name or "gemma" in name or "phi" in name:
        # Decoder-only models—common attention proj names
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Fallback: try common names; you can override via CLI
    return ["query", "key", "value", "dense"]


def compute_metrics(predictions, labels, label_names):
    preds = np.argmax(predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    cm = confusion_matrix(labels, preds).tolist()
    cls_report = classification_report(
        labels, preds, target_names=label_names, digits=4
    )
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": cls_report,
    }


def plot_confusion_matrix(cm, labels, output_path):
    cm = np.array(cm)
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


def safe_training_args(**kwargs):
    """Strip unsupported TrainingArguments depending on transformers version, and keep strategies consistent."""
    sig = inspect.signature(TrainingArguments.__init__)
    valid = sig.parameters.keys()

    supports_eval = "evaluation_strategy" in valid
    supports_save = "save_strategy" in valid
    supports_load_best = "load_best_model_at_end" in valid
    supports_metric_for_best = "metric_for_best_model" in valid

    # If eval strategy isn't supported, make sure we don't pass anything that depends on it
    if not supports_eval:
        kwargs.pop("evaluation_strategy", None)
        kwargs.pop("save_strategy", None)          # avoid mismatch
        kwargs.pop("load_best_model_at_end", None) # avoid mismatch
        kwargs.pop("metric_for_best_model", None)

    # Clean kwargs to only those supported
    clean = {k: v for k, v in kwargs.items() if k in valid and v is not None}
    return TrainingArguments(**clean)



def run_eval(model, tokenizer, df, label_names, max_len):
    ds = AbstractDataset(df, tokenizer, max_len)
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for item in ds:
            input_ids = item["input_ids"].unsqueeze(0)
            attn = item["attention_mask"].unsqueeze(0)
            label = item["labels"].item()
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            preds.append(logits.detach().cpu().numpy())
            labels.append(label)
    preds = np.concatenate(preds, axis=0)
    return compute_metrics(preds, labels, label_names)


# -----------------------------
# Main
# -----------------------------
def main(args):
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "label_map.json", "r") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = {k: int(v) for k, v in label_map["label2id"].items()}
    label_names = [id2label[i] for i in sorted(id2label.keys())]

    # Tokenizer + base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    if args.use_4bit:
        model, is_4bit = maybe_cast_bnb(model, True)
        if not is_4bit:
            args.use_4bit = False  # fallback

    # Determine target modules for LoRA
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",")]
    else:
        target_modules = select_default_target_modules(args.model_name)
    print(f"[Info] Using LoRA target modules: {target_modules}")

    # Build LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="SEQ_CLS",
    )

    # Wrap model with LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = AbstractDataset(train_df, tokenizer, args.max_len)
    val_dataset = AbstractDataset(val_df, tokenizer, args.max_len)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training args (version-safe)
    training_args = safe_training_args(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        gradient_accumulation_steps=args.grad_accum_steps,
        fp16=False,  # keeps it CPU safe by default; change if you have GPU
        bf16=False,
    )

    def hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_metrics(logits, labels, label_names)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=hf_compute_metrics,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate on validation (best model already loaded if supported)
    val_metrics = trainer.evaluate()
    with open(output_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    # Evaluate on test set manually (since Trainer only knows eval_dataset)
    test_metrics = run_eval(model, tokenizer, test_df, label_names, args.max_len)
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Confusion matrix plot
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        label_names,
        output_dir / "confusion_matrix_test.png",
    )

    print("\n[Done] LoRA fine-tuned model saved to:", output_dir)
    print("[Val metrics]", json.dumps(val_metrics, indent=2))
    print("[Test metrics]", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for cancer vs non-cancer classification"
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Try 'distilbert-base-uncased' (CPU), 'microsoft/phi-2', 'google/gemma-2b-it' (GPU)")
    parser.add_argument("--output_dir", type=str, default="models/finetuned_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default=None,
                        help="Comma-separated module names. If not set, picks defaults.")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (QLoRA). Requires bitsandbytes & GPU.")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    args = parser.parse_args()

    main(args)
