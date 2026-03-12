"""
phobert_classifier.py - Model class definitions.

PhoBERTClassifier:          PhoBERT + linear head (for topic classification)
PhoBERTWithTopicFeatures:   PhoBERT + topic logits concat → MLP head (for sentiment)
"""

import torch.nn as nn
from transformers import AutoModel


class PhoBERTClassifier(nn.Module):
    """PhoBERT encoder + classification head."""

    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class PhoBERTWithTopicFeatures(nn.Module):
    """PhoBERT encoder + topic logits concatenated to [CLS] → MLP head."""

    def __init__(self, model_name, num_classes, topic_dim=4, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + topic_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, topic_features=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        cls_output = self.dropout(cls_output)
        if topic_features is not None:
            import torch
            cls_output = torch.cat([cls_output, topic_features], dim=1)
        logits = self.classifier(cls_output)
        return logits
