# 📝 Student Feedback Sentiment Analysis

Phân loại cảm xúc phản hồi sinh viên sử dụng **PhoBERT** (SOTA cho NLP tiếng Việt).

## 📁 Cấu trúc dự án

```
Sentiment-Analysis/
├── config.yaml               # Cấu hình hyperparameters, paths
├── requirements.txt           # Thư viện cần thiết
├── PROBLEM.md                 # Mô tả đề bài
├── README.md                  # Hướng dẫn sử dụng
├── data/
│   ├── train.csv              # Dữ liệu huấn luyện
│   ├── test.csv               # Dữ liệu kiểm tra
│   └── sample_submission.csv
├── model/                     # Định nghĩa model & load checkpoint
│   ├── __init__.py
│   ├── phobert_classifier.py  # PhoBERTClassifier, PhoBERTWithTopicFeatures
│   └── loader.py              # save_checkpoint, load_topic/sentiment_model
├── src/
│   ├── utils.py               # Data, preprocessing, dataset, training loops
│   ├── train_topic.py         # Stage 1: Train topic model
│   ├── train_sentiment.py     # Stage 2: Train sentiment model
│   └── main.py                # Tạo file submission
├── checkpoint/                # (tự tạo) Model weights đã train
└── output/                    # (tự tạo) File submission.csv
```

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Chạy pipeline

```bash
# Stage 1: Train topic model
python src/train_topic.py

# Stage 2: Train sentiment model (dùng topic probs từ Stage 1)
python src/train_sentiment.py

# Tạo submission
python src/main.py
```

### ⚡ Chạy nhanh

```bash
python src/train_topic.py && python src/train_sentiment.py && python src/main.py
```

## 🏗️ Kiến trúc

```
sentence → PhoBERT → [CLS] → Topic Head → topic_probs (4-dim)
                                                ↓
sentence → PhoBERT → [CLS] ── concat ── topic_probs
                                  ↓
                          MLP Head → sentiment (0/1/2)
```

**Pipeline 2 giai đoạn:**
1. **Topic Model** (`model/phobert_classifier.py → PhoBERTClassifier`)
2. **Sentiment Model** (`model/phobert_classifier.py → PhoBERTWithTopicFeatures`)

| Điểm chính | Mô tả |
|-------------|--------|
| Model | PhoBERT-base (vinai/phobert-base) |
| OOF | Out-of-Fold topic predictions tránh data leakage |
| Word Seg | underthesea |
| Checkpoint | Lưu vào `checkpoint/` |

## 📈 Kết quả thực nghiệm

| Model | Approach | Avg Fold F1 | Overall OOF F1 |
|-------|----------|-------------|-----------------|
| Topic (Stage 1) | TF-IDF + LightGBM | 0.7178 | 0.7183 |
| Sentiment (Stage 2) | TF-IDF + LightGBM | 0.7096 | 0.7099 |
| Topic (Stage 1) | **PhoBERT** | **0.8056** | **0.8058** |
| Sentiment (Stage 2) | **PhoBERT** | **0.8373** | **0.8377** |

## ⚙️ Cấu hình (`config.yaml`)

| Tham số | Mô tả | Mặc định |
|---------|--------|----------|
| `phobert.model_name` | Pre-trained model | `vinai/phobert-base` |
| `phobert.max_length` | Max token length | 128 |
| `phobert.batch_size` | Batch size | 32 |
| `topic_model.epochs` | Epochs Stage 1 | 5 |
| `sentiment_model.epochs` | Epochs Stage 2 | 5 |
| `cv.n_splits` | K-Fold | 5 |

## 📝 Nhãn

| Sentiment | Ý nghĩa | Topic | Ý nghĩa |
|-----------|----------|-------|----------|
| 0 | Tiêu cực | 0 | Giảng viên |
| 1 | Trung tính | 1 | Chương trình đào tạo |
| 2 | Tích cực | 2 | Cơ sở vật chất |
| | | 3 | Khác |
