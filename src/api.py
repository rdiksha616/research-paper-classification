#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
import uvicorn
import json
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from disease_extraction import extract_diseases_rulebased as extract_diseases

# ---------------------------
# Load Model & Tokenizer
# ---------------------------
MODEL_DIR = Path("models/finetuned_lora")
DATA_DIR = Path("data/processed")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(DATA_DIR / "label_map.json", "r") as f:
    label_map = json.load(f)
id2label = {int(v): k for k, v in label_map["label2id"].items()}


def predict(text, max_len=256):
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
    pred_idx = int(probs.argmax())
    predicted_label = id2label[pred_idx]
    confidence_scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    diseases = extract_diseases(text)
    return {
        "predicted_labels": [predicted_label],
        "confidence_scores": confidence_scores,
        "extracted_diseases": diseases
    }


# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="Research Paper Classifier API")


class PredictRequest(BaseModel):
    text: str


@app.post("/predict-text")
def predict_text(req: PredictRequest):
    result = predict(req.text)
    return {"abstract_id": "manual_input", **result}


@app.post("/predict-file")
async def predict_file(file: UploadFile):
    df = pd.read_csv(file.file)
    results = []
    for _, row in df.iterrows():
        abstract_id = row.get("abstract_id", f"id_{_}")
        text = row["text"]
        result = predict(text)
        results.append({"abstract_id": abstract_id, **result})
    return results


@app.get("/")
def root():
    return {"message": "Research Paper Classifier API is running."}


# For local testing: python src/api.py
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
