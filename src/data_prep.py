#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def read_txt_files(root: Path, label: str) -> List[dict]:
    rows = []
    for p in sorted(root.glob("*.txt")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            text = ""
        rows.append(
            {
                "abstract_id": p.stem,  # pubmed id (filename without .txt)
                "text": text,
                "label": label,
                "filepath": str(p),
            }
        )
    return rows


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove bracketed numeric citations like [1], [12], [1,2,3]
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", " ", text)
    # Remove parenthesis-only years like (2018)
    text = re.sub(r"\(\s*20\d{2}\s*\)", " ", text)
    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_df(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1

    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {
        "num_samples": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "avg_len_chars": float(df["text"].str.len().mean()),
        "avg_len_words": float(df["text"].str.split().map(len).mean()),
        "min_len_words": int(df["text"].str.split().map(len).min()),
        "max_len_words": int(df["text"].str.split().map(len).max()),
    }
    return stats


def main(args):
    raw_root = Path(args.dataset_root).resolve()
    cancer_dir = raw_root / "Cancer"
    non_cancer_dir = raw_root / "Non-Cancer"

    if not cancer_dir.exists() or not non_cancer_dir.exists():
        raise FileNotFoundError(
            f"Expected directories 'Cancer' and 'Non-Cancer' under: {raw_root}"
        )

    out_dir = Path(args.out_dir).resolve()
    (out_dir / "processed").mkdir(parents=True, exist_ok=True)

    print("[1/5] Reading raw txt files ...")
    rows = []
    rows += read_txt_files(cancer_dir, "Cancer")
    rows += read_txt_files(non_cancer_dir, "Non-Cancer")

    print(f"   - Total files read: {len(rows)}")

    print("[2/5] Creating DataFrame & cleaning ...")
    df = pd.DataFrame(rows)
    df["text"] = df["text"].map(basic_clean)

    # Drop empty / missing abstracts
    before = len(df)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"   - Dropped {dropped} rows with empty text")

    # Map labels to ids
    labels = sorted(df["label"].unique())
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    # Shuffle consistently
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Save the full cleaned dataset
    full_csv = out_dir / "processed" / "abstracts_full.csv"
    df.to_csv(full_csv, index=False)
    print(f"[3/5] Saved cleaned full dataset: {full_csv}")

    # Train/Val/Test split
    print("[4/5] Splitting into train/val/test ...")
    train_df, val_df, test_df = split_df(
        df, args.train_ratio, args.val_ratio, args.seed
    )

    for split_name, split_df_ in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        split_path = out_dir / "processed" / f"{split_name}.csv"
        split_df_.to_csv(split_path, index=False)
        print(f"   - {split_name}: {len(split_df_)} ? {split_path}")

    # Save label map
    label_map_path = out_dir / "processed" / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    print(f"[5/5] Saved label maps: {label_map_path}")

    # Basic stats
    stats = {
        "global": compute_stats(df),
        "train": compute_stats(train_df),
        "val": compute_stats(val_df),
        "test": compute_stats(test_df),
    }
    stats_path = out_dir / "processed" / "eda_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[Done] Saved EDA stats: {stats_path}")

    print("\nSummary:")
    print(json.dumps(stats, indent=2))
    print("\nLabel map:", label2id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare cancer vs non-cancer abstracts dataset"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the extracted Dataset folder containing Cancer/ and Non-Cancer/",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data",
        help="Where to save processed CSVs & maps",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
