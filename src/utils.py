"""
utils.py - Data loading, text preprocessing, Dataset, training loops.
Model definitions are in model/ package.
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report


# ============================================================
# Emoticon mappings
# ============================================================
EMOTICON_MAP = {
    "colonsmile": " emoji_smile ",
    "colonsad": " emoji_sad ",
    "colonsurprise": " emoji_surprise ",
    "colonlove": " emoji_love ",
    "colonsmilesmile": " emoji_happy ",
    "coloncontemn": " emoji_contemn ",
    "colonbigsmile": " emoji_bigsmile ",
    "coloncc": " emoji_cc ",
    "colonsmallsmile": " emoji_smallsmile ",
    "coloncolon": " emoji_colon ",
    "colonlovelove": " emoji_lovelove ",
    "colonhihi": " emoji_hihi ",
    "colonsadcolon": " emoji_cry ",
    "colondoublesurprise": " emoji_doublesurprise ",
    "vdotv": " emoji_sad ",
    "dotdotdot": " ... ",
    "doubledot": " : ",
    "fraction": " / ",
}


# ============================================================
# Config & Paths
# ============================================================

def load_config(config_path="config.yaml"):
    if not os.path.isabs(config_path):
        config_path = os.path.join(get_project_root(), config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dirs(config):
    root = get_project_root()
    for key in ["checkpoint_dir", "output_dir"]:
        os.makedirs(os.path.join(root, config["paths"][key]), exist_ok=True)


# ============================================================
# Data Loading
# ============================================================

def load_data(config):
    root = get_project_root()
    data_dir = os.path.join(root, config["paths"]["data_dir"])
    train_df = pd.read_csv(os.path.join(data_dir, config["paths"]["train_file"]))
    test_df = pd.read_csv(os.path.join(data_dir, config["paths"]["test_file"]))
    print(f"[DATA] Train: {train_df.shape} | Test: {test_df.shape}")
    return train_df, test_df


# ============================================================
# Text Preprocessing
# ============================================================

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    for enc, rep in EMOTICON_MAP.items():
        text = text.replace(enc, rep)
    text = re.sub(r"wzjwz\d+", "", text)
    text = re.sub(r"\w+dot\w+dot\w+dot\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def try_word_segment(texts):
    try:
        from underthesea import word_tokenize
        print("[INFO] Word segmentation (underthesea)...")
        result = []
        for i, t in enumerate(texts):
            if i % 3000 == 0 and i > 0:
                print(f"  {i}/{len(texts)}")
            try:
                result.append(word_tokenize(t, format="text"))
            except Exception:
                result.append(t)
        print(f"  {len(texts)}/{len(texts)} Done!")
        return result
    except ImportError:
        print("[WARN] underthesea not found, skipping.")
        return list(texts)


def preprocess_series(series):
    """Preprocess + word segment a pandas Series of texts."""
    cleaned = series.apply(preprocess_text).tolist()
    return try_word_segment(cleaned)


# ============================================================
# PyTorch Dataset
# ============================================================

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128, topic_features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.topic_features = topic_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.topic_features is not None:
            item["topic_features"] = torch.tensor(self.topic_features[idx], dtype=torch.float32)
        return item


# ============================================================
# Training & Evaluation Loops
# ============================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        kwargs = {}
        if "topic_features" in batch:
            kwargs["topic_features"] = batch["topic_features"].to(device)

        logits = model(input_ids, attention_mask, **kwargs)
        loss = criterion(logits, labels) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        kwargs = {}
        if "topic_features" in batch:
            kwargs["topic_features"] = batch["topic_features"].to(device)

        logits = model(input_ids, attention_mask, **kwargs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        if "labels" in batch:
            all_labels.append(batch["labels"].numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels) if all_labels else None
    return all_preds, all_probs, all_labels


def evaluate_model(y_true, y_pred, label_names=None):
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"  Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))
    return f1


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
