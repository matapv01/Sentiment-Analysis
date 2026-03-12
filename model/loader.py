"""
loader.py - Save/load model checkpoints and tokenizer.
"""

import os
import torch
from transformers import AutoTokenizer

from model.phobert_classifier import PhoBERTClassifier, PhoBERTWithTopicFeatures


def load_tokenizer(model_name):
    """Load PhoBERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[LOAD] Tokenizer: {model_name}")
    return tokenizer


def save_checkpoint(model, path):
    """Save model state dict to checkpoint path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[SAVE] Checkpoint -> {path}")


def load_topic_model(model_name, num_classes, checkpoint_path, dropout=0.1, device="cpu"):
    """Load a trained PhoBERTClassifier for topic prediction."""
    model = PhoBERTClassifier(model_name, num_classes, dropout)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[LOAD] Topic model <- {checkpoint_path}")
    return model


def load_sentiment_model(model_name, num_classes, checkpoint_path, topic_dim=4, dropout=0.1, device="cpu"):
    """Load a trained PhoBERTWithTopicFeatures for sentiment prediction."""
    model = PhoBERTWithTopicFeatures(model_name, num_classes, topic_dim, dropout)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[LOAD] Sentiment model <- {checkpoint_path}")
    return model


def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Generic: load state dict into an existing model."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[LOAD] Checkpoint <- {checkpoint_path}")
    return model
