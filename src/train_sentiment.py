"""
train_sentiment.py - Stage 2: Fine-tune PhoBERT for Sentiment Classification.

Uses topic probabilities from Stage 1 as additional features.

Saves:
  - checkpoint/sentiment_fold{i}.pt        (model weights per fold)
  - checkpoint/sentiment_test_preds.npy    (final test predictions)
"""

import os
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, load_data, ensure_dirs, get_project_root,
    preprocess_series, FeedbackDataset,
    train_epoch, eval_epoch, evaluate_model, set_seed,
)
from model import PhoBERTWithTopicFeatures, save_checkpoint, load_tokenizer


def train_sentiment_model(config):
    root = get_project_root()
    ensure_dirs(config)
    seed = config["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    ckpt_dir = os.path.join(root, config["paths"]["checkpoint_dir"])
    model_name = config["phobert"]["model_name"]
    max_length = config["phobert"]["max_length"]
    batch_size = config["phobert"]["batch_size"]
    accum_steps = config["phobert"]["accumulation_steps"]

    # ---- Data ----
    print("=" * 60)
    print("STAGE 2: SENTIMENT MODEL (PhoBERT + Topic Features)")
    print("=" * 60)
    train_df, test_df = load_data(config)
    train_texts = preprocess_series(train_df["sentence"])
    test_texts = preprocess_series(test_df["sentence"])

    tokenizer = load_tokenizer(model_name)
    y_sentiment = train_df["sentiment"].values
    n_classes = config["sentiment_model"]["num_class"]
    n_splits = config["cv"]["n_splits"]
    topic_dim = config["sentiment_model"]["topic_feature_dim"]

    # ---- Load topic probs from Stage 1 ----
    oof_topic_path = os.path.join(ckpt_dir, "topic_oof_probs.npy")
    test_topic_path = os.path.join(ckpt_dir, "topic_test_probs.npy")

    if not os.path.exists(oof_topic_path):
        print("[ERROR] Topic probs not found! Run train_topic.py first.")
        sys.exit(1)

    oof_topic_probs = np.load(oof_topic_path)
    test_topic_probs = np.load(test_topic_path)
    print(f"[INFO] OOF topic probs: {oof_topic_probs.shape}")
    print(f"[INFO] Test topic probs: {test_topic_probs.shape}")

    # ---- CV ----
    skf = StratifiedKFold(n_splits=n_splits, shuffle=config["cv"]["shuffle"], random_state=seed)
    oof_preds = np.zeros(len(train_df), dtype=int)
    test_probs = np.zeros((len(test_df), n_classes))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, y_sentiment)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1}/{n_splits}")
        print(f"{'='*60}")

        tr_texts = [train_texts[i] for i in train_idx]
        val_texts = [train_texts[i] for i in val_idx]
        tr_labels = y_sentiment[train_idx]
        val_labels = y_sentiment[val_idx]
        tr_topic = oof_topic_probs[train_idx]
        val_topic = oof_topic_probs[val_idx]

        train_ds = FeedbackDataset(tr_texts, tr_labels, tokenizer, max_length, tr_topic)
        val_ds = FeedbackDataset(val_texts, val_labels, tokenizer, max_length, val_topic)
        test_ds = FeedbackDataset(test_texts, tokenizer=tokenizer, max_length=max_length, topic_features=test_topic_probs)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

        # Model
        model = PhoBERTWithTopicFeatures(
            model_name, n_classes,
            topic_dim=topic_dim,
            dropout=config["sentiment_model"]["dropout"],
        ).to(device)

        # Optimizer & Scheduler
        epochs = config["sentiment_model"]["epochs"]
        total_steps = len(train_dl) * epochs // accum_steps
        warmup_steps = int(total_steps * config["sentiment_model"]["warmup_ratio"])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["sentiment_model"]["learning_rate"],
            weight_decay=config["sentiment_model"]["weight_decay"],
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_f1 = 0
        best_path = os.path.join(ckpt_dir, f"sentiment_fold{fold}.pt")

        for epoch in range(epochs):
            loss = train_epoch(model, train_dl, optimizer, scheduler, device, accum_steps)
            val_preds_ep, val_probs_ep, val_labels_np = eval_epoch(model, val_dl, device)
            f1 = evaluate_model(
                val_labels_np, val_preds_ep,
                label_names=["Negative", "Neutral", "Positive"]
            )
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(model, best_path)

        # Load best & predict
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        model.eval()

        val_preds_best, _, _ = eval_epoch(model, val_dl, device)
        oof_preds[val_idx] = val_preds_best

        _, test_probs_fold, _ = eval_epoch(model, test_dl, device)
        test_probs += test_probs_fold / n_splits

        fold_f1s.append(best_f1)
        print(f"  Fold {fold+1} Best F1: {best_f1:.4f}")

    # ---- OOF Summary ----
    print("\n" + "=" * 60)
    overall_f1 = evaluate_model(
        y_sentiment, oof_preds,
        label_names=["Negative", "Neutral", "Positive"]
    )
    print(f"[RESULT] Avg Fold F1: {np.mean(fold_f1s):.4f}")
    print(f"[RESULT] Overall OOF F1: {overall_f1:.4f}")

    # Save test predictions
    test_preds = np.argmax(test_probs, axis=1)
    np.save(os.path.join(ckpt_dir, "sentiment_test_preds.npy"), test_preds)
    np.save(os.path.join(ckpt_dir, "sentiment_test_probs.npy"), test_probs)
    print(f"[SAVE] sentiment_test_preds/probs.npy -> {ckpt_dir}")
    print("\n[DONE] Stage 2 complete!")


if __name__ == "__main__":
    config = load_config()
    train_sentiment_model(config)
