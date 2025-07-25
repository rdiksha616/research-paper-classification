#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)

def pick(d, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default

def extract_core(m):
    # Try common variants used by HF Trainer / our scripts
    acc = pick(m, "accuracy", "eval_accuracy")
    f1  = pick(m, "f1", "f1_score", "eval_f1")
    prec = pick(m, "precision", "eval_precision")
    rec  = pick(m, "recall", "eval_recall")
    cm   = pick(m, "confusion_matrix", "eval_confusion_matrix")
    cls_rep = pick(m, "classification_report", "report")

    # Some HF eval dicts wrap metrics under 'metrics'
    if acc is None and "metrics" in m:
        return extract_core(m["metrics"])

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "classification_report": cls_rep,
    }

def flat_cm_stats(cm):
    if not cm or len(cm) != 2 or len(cm[0]) != 2:
        return None
    # [[TN, FP], [FN, TP]] OR sometimes [[TP, FN], [FP, TN]]
    # Our pipeline used sklearn -> format is:
    # rows = true labels (0,1), cols = predicted (0,1)
    # So:
    # TN = cm[0][0], FP = cm[0][1], FN = cm[1][0], TP = cm[1][1]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return tn, fp, fn, tp

def print_report(base, ft):
    print("\n===== Model Comparison Report =====")
    print(f"Baseline Accuracy   : {base['accuracy']:.4f}")
    print(f"Fine-tuned Accuracy : {ft['accuracy']:.4f}")
    print(f"Δ Accuracy          : {ft['accuracy'] - base['accuracy']:+.4f}")

    if base["f1"] is not None and ft["f1"] is not None:
        print(f"\nBaseline F1-score   : {base['f1']:.4f}")
        print(f"Fine-tuned F1-score : {ft['f1']:.4f}")
        print(f"Δ F1-score          : {ft['f1'] - base['f1']:+.4f}")

    print("\nBaseline Confusion Matrix:")
    print(base["confusion_matrix"])
    print("Fine-tuned Confusion Matrix:")
    print(ft["confusion_matrix"])

    base_stats = flat_cm_stats(base["confusion_matrix"])
    ft_stats   = flat_cm_stats(ft["confusion_matrix"])
    if base_stats and ft_stats:
        btn, bfp, bfn, btp = base_stats
        ftn, ffp, ffn, ftp = ft_stats
        print("\nΔ FP (Fine - Base):", ffp - bfp)
        print("Δ FN (Fine - Base):", ffn - bfn)

    if base["classification_report"]:
        print("\nBaseline Classification Report:\n", base["classification_report"])
    if ft["classification_report"]:
        print("\nFine-tuned Classification Report:\n", ft["classification_report"])

def main():
    baseline_path  = Path("models/baseline/test_metrics.json")
    finetuned_path = Path("models/finetuned_lora/test_metrics.json")

    if not baseline_path.exists():
        print("[Error] models/baseline/test_metrics.json not found. Run evaluate.py for the baseline model.")
        return
    if not finetuned_path.exists():
        print("[Error] models/finetuned_lora/test_metrics.json not found. Finish LoRA fine-tuning & eval first.")
        return

    baseline_raw  = load_metrics(baseline_path)
    finetuned_raw = load_metrics(finetuned_path)

    base = extract_core(baseline_raw)
    ft   = extract_core(finetuned_raw)

    missing = [k for k in ["accuracy","f1","confusion_matrix"] if base.get(k) is None or ft.get(k) is None]
    if missing:
        print("[Warn] Some metrics are missing:", missing)
        print("Re-run evaluate.py on both models to regenerate standardized test_metrics.json")
    print_report(base, ft)

if __name__ == "__main__":
    main()
