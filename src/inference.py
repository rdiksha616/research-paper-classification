#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from disease_extraction import extract_diseases_rulebased as extract_diseases


def predict_single(model, tokenizer, text, label_map, max_len=256):
    """
    Predict class & confidence for a single abstract.
    """
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1).numpy().flatten()

    id2label = {int(v): k for k, v in label_map["label2id"].items()}
    pred_idx = int(probs.argmax())
    predicted_label = id2label[pred_idx]
    confidence_scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    return predicted_label, confidence_scores


def main(args):
    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    with open(Path(args.data_dir) / "label_map.json", "r") as f:
        label_map = json.load(f)

    results = []

    if args.input_text:
        text = args.input_text
        predicted_label, confidence = predict_single(model, tokenizer, text, label_map)
        diseases = extract_diseases(text)
        results.append({
            "abstract_id": "manual_input",
            "predicted_labels": [predicted_label],
            "confidence_scores": confidence,
            "extracted_diseases": diseases
        })

    elif args.input_csv:
        df = pd.read_csv(args.input_csv)
        for _, row in df.iterrows():
            abstract_id = row.get("abstract_id", f"id_{_}")
            text = row["text"]
            predicted_label, confidence = predict_single(model, tokenizer, text, label_map)
            diseases = extract_diseases(text)
            results.append({
                "abstract_id": abstract_id,
                "predicted_labels": [predicted_label],
                "confidence_scores": confidence,
                "extracted_diseases": diseases
            })

    out_path = Path(args.output_json)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Done] Predictions saved to: {out_path}")
    print(json.dumps(results[:3], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with fine-tuned LoRA model")
    parser.add_argument("--model_dir", type=str, default="models/finetuned_lora")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--input_text", type=str, help="Single abstract text")
    parser.add_argument("--input_csv", type=str, help="CSV file with 'text' column")
    parser.add_argument("--output_json", type=str, default="predictions.json")
    args = parser.parse_args()
    main(args)
