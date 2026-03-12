"""
main.py - Inference: Load checkpoints, predict topic → sentiment, generate submission.csv
"""

import os
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, ensure_dirs, get_project_root,
    preprocess_series, FeedbackDataset, eval_epoch, set_seed,
)
from model import PhoBERTClassifier, PhoBERTWithTopicFeatures, load_tokenizer
from model.loader import load_topic_model, load_sentiment_model


def main():
    print("=" * 60)
    print("INFERENCE: GENERATING SUBMISSION")
    print("=" * 60)

    config = load_config()
    root = get_project_root()
    ensure_dirs(config)
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    ckpt_dir = os.path.join(root, config["paths"]["checkpoint_dir"])
    model_name = config["phobert"]["model_name"]
    max_length = config["phobert"]["max_length"]
    batch_size = config["phobert"]["batch_size"]
    n_splits = config["cv"]["n_splits"]

    # ---- Check for pre-computed predictions (fast path) ----
    precomputed = os.path.join(ckpt_dir, "sentiment_test_preds.npy")
    if os.path.exists(precomputed):
        print("[INFO] Found pre-computed predictions from training.")
        test_preds = np.load(precomputed)
    else:
        # ---- Full inference ----
        data_dir = os.path.join(root, config["paths"]["data_dir"])
        test_df = pd.read_csv(os.path.join(data_dir, config["paths"]["test_file"]))
        test_texts = preprocess_series(test_df["sentence"])
        tokenizer = load_tokenizer(model_name)

        # Stage 1: Topic predictions
        print("\n[INFO] Stage 1: Predicting topics...")
        n_topic = config["topic_model"]["num_class"]
        topic_probs = np.zeros((len(test_df), n_topic))

        for fold in range(n_splits):
            ckpt_path = os.path.join(ckpt_dir, f"topic_fold{fold}.pt")
            model = load_topic_model(
                model_name, n_topic, ckpt_path,
                dropout=config["topic_model"]["dropout"], device=device,
            )
            ds = FeedbackDataset(test_texts, tokenizer=tokenizer, max_length=max_length)
            dl = DataLoader(ds, batch_size=batch_size * 2, shuffle=False, num_workers=2)
            _, probs, _ = eval_epoch(model, dl, device)
            topic_probs += probs / n_splits
            del model
            torch.cuda.empty_cache()

        # Stage 2: Sentiment predictions
        print("\n[INFO] Stage 2: Predicting sentiment...")
        n_sent = config["sentiment_model"]["num_class"]
        topic_dim = config["sentiment_model"]["topic_feature_dim"]
        sent_probs = np.zeros((len(test_df), n_sent))

        for fold in range(n_splits):
            ckpt_path = os.path.join(ckpt_dir, f"sentiment_fold{fold}.pt")
            model = load_sentiment_model(
                model_name, n_sent, ckpt_path,
                topic_dim=topic_dim,
                dropout=config["sentiment_model"]["dropout"], device=device,
            )
            ds = FeedbackDataset(
                test_texts, tokenizer=tokenizer, max_length=max_length,
                topic_features=topic_probs,
            )
            dl = DataLoader(ds, batch_size=batch_size * 2, shuffle=False, num_workers=2)
            _, probs, _ = eval_epoch(model, dl, device)
            sent_probs += probs / n_splits
            del model
            torch.cuda.empty_cache()

        test_preds = np.argmax(sent_probs, axis=1)

    # ---- Generate submission ----
    data_dir = os.path.join(root, config["paths"]["data_dir"])
    test_df = pd.read_csv(os.path.join(data_dir, config["paths"]["test_file"]))
    submission = pd.DataFrame({"id": test_df["id"], "sentiment": test_preds.astype(int)})

    out_path = os.path.join(root, config["paths"]["output_dir"], config["paths"]["submission_file"])
    submission.to_csv(out_path, index=False)

    print(f"\n[SAVE] {out_path}")
    print(f"[INFO] Shape: {submission.shape}")
    print(f"[INFO] Distribution:")
    for val, name in {0: "Negative", 1: "Neutral", 2: "Positive"}.items():
        cnt = (test_preds == val).sum()
        print(f"  {name}: {cnt} ({cnt/len(test_preds)*100:.1f}%)")
    print(submission.head(10).to_string(index=False))
    print("\n[DONE] Submission ready!")


if __name__ == "__main__":
    main()
