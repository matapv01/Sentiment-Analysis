"""
train_topic.py - Stage 1: Fine-tune PhoBERT for Topic Classification.

Saves:
  - checkpoint/topic_fold{i}.pt        (model weights per fold)
  - checkpoint/topic_oof_probs.npy     (OOF topic probabilities)
  - checkpoint/topic_test_probs.npy    (averaged test topic probabilities)
"""

import os
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

# Add project root to sys.path so we can import model/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config, load_data, ensure_dirs, get_project_root,
    preprocess_series, FeedbackDataset,
    train_epoch, eval_epoch, evaluate_model, set_seed,
)
from model import PhoBERTClassifier, save_checkpoint, load_tokenizer


def train_topic_model(config):
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
    print("STAGE 1: TOPIC MODEL (PhoBERT)")
    print("=" * 60)
    train_df, test_df = load_data(config)
    train_texts = preprocess_series(train_df["sentence"])
    test_texts = preprocess_series(test_df["sentence"])

    tokenizer = load_tokenizer(model_name)
    y_topic = train_df["topic"].values
    n_classes = config["topic_model"]["num_class"]
    n_splits = config["cv"]["n_splits"]

    # ---- CV ----
    skf = StratifiedKFold(n_splits=n_splits, shuffle=config["cv"]["shuffle"], random_state=seed)
    oof_probs = np.zeros((len(train_df), n_classes))
    test_probs = np.zeros((len(test_df), n_classes))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, y_topic)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1}/{n_splits}")
        print(f"{'='*60}")

        tr_texts = [train_texts[i] for i in train_idx]
        val_texts = [train_texts[i] for i in val_idx]
        tr_labels = y_topic[train_idx]
        val_labels = y_topic[val_idx]

        train_ds = FeedbackDataset(tr_texts, tr_labels, tokenizer, max_length)
        val_ds = FeedbackDataset(val_texts, val_labels, tokenizer, max_length)
        test_ds = FeedbackDataset(test_texts, tokenizer=tokenizer, max_length=max_length)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

        # Model
        model = PhoBERTClassifier(
            model_name, n_classes, dropout=config["topic_model"]["dropout"]
        ).to(device)

        # Optimizer & Scheduler
        epochs = config["topic_model"]["epochs"]
        total_steps = len(train_dl) * epochs // accum_steps
        warmup_steps = int(total_steps * config["topic_model"]["warmup_ratio"])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["topic_model"]["learning_rate"],
            weight_decay=config["topic_model"]["weight_decay"],
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_f1 = 0
        best_path = os.path.join(ckpt_dir, f"topic_fold{fold}.pt")

        for epoch in range(epochs):
            loss = train_epoch(model, train_dl, optimizer, scheduler, device, accum_steps)
            val_preds, val_probs, val_labels_np = eval_epoch(model, val_dl, device)
            f1 = evaluate_model(
                val_labels_np, val_preds,
                label_names=["Lecturer", "Training", "Facility", "Others"]
            )
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                save_checkpoint(model, best_path)

        # Load best & predict
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        model.eval()

        _, val_probs_best, _ = eval_epoch(model, val_dl, device)
        oof_probs[val_idx] = val_probs_best

        _, test_probs_fold, _ = eval_epoch(model, test_dl, device)
        test_probs += test_probs_fold / n_splits

        fold_f1s.append(best_f1)
        print(f"  Fold {fold+1} Best F1: {best_f1:.4f}")

    # ---- OOF Summary ----
    print("\n" + "=" * 60)
    oof_preds = np.argmax(oof_probs, axis=1)
    overall_f1 = evaluate_model(
        y_topic, oof_preds,
        label_names=["Lecturer", "Training", "Facility", "Others"]
    )
    print(f"[RESULT] Avg Fold F1: {np.mean(fold_f1s):.4f}")
    print(f"[RESULT] Overall OOF F1: {overall_f1:.4f}")

    # Save OOF & test probs
    np.save(os.path.join(ckpt_dir, "topic_oof_probs.npy"), oof_probs)
    np.save(os.path.join(ckpt_dir, "topic_test_probs.npy"), test_probs)
    print(f"[SAVE] topic_oof_probs.npy, topic_test_probs.npy -> {ckpt_dir}")
    print("\n[DONE] Stage 1 complete!")


if __name__ == "__main__":
    config = load_config()
    train_topic_model(config)
