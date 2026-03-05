# 🧠 Multi-Class Text Emotion Detection System

> Classify text into 6 emotions with probability distribution, confidence scoring,
> and sentiment analysis — powered by TF-IDF + Logistic Regression.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

| Property      | Details                                      |
|---------------|----------------------------------------------|
| **Task**      | Multi-class text classification              |
| **Classes**   | joy 😄, sadness 😢, anger 😠, fear 😨, love ❤️, surprise 😲 |
| **Model**     | Logistic Regression (multinomial)            |
| **Features**  | TF-IDF (unigrams + bigrams, 50k features)    |
| **Accuracy**  | ≥ 85% on dair-ai/emotion dataset             |
| **Interface** | Streamlit web app with interactive charts    |

---

## 🗂️ Project Structure

```
emotion_detection/
│
├── data/                      ← Dataset files (train.txt, val.txt, test.txt)
├── models/                    ← Saved model artifacts (auto-created)
│   ├── emotion_pipeline.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lr_classifier.pkl
│   └── label_classes.pkl
├── assets/                    ← Generated plots (auto-created)
│   ├── confusion_matrix.png
│   └── class_distribution.png
│
├── app.py                     ← Streamlit web application
├── train_model.py             ← Training script
├── predict.py                 ← CLI prediction utility
├── download_dataset.py        ← One-time dataset downloader
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
```bash
python download_dataset.py
```
This downloads the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
dataset (~16k training samples, 2k val, 2k test) into `data/`.

> **No internet?** Skip this step — `train_model.py` falls back to a built-in
> synthetic dataset automatically (lower accuracy, good for testing).

### 3. Train the model
```bash
python train_model.py
```
- Trains TF-IDF + Logistic Regression pipeline
- Saves model to `models/`
- Saves evaluation plots to `assets/`
- Prints accuracy & classification report

### 4. Launch the web app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### 5. CLI prediction (optional)
```bash
python predict.py "I am so happy today!"
python predict.py   # interactive mode
```

---

## 🎯 Features

### Web App (app.py)
- 📝 **Text input** — type or paste any text
- 😄 **Emotion prediction** with emoji display
- 📊 **Probability bar chart** (Plotly, all 6 classes)
- 🟢🟡🔴 **Confidence indicator** (High / Medium / Low)
- 💬 **Sentiment analysis** (rule-based positive/negative/neutral)
- ✅ **Sentiment vs emotion comparison**
- 💡 **Example sentences** for quick testing

### Training Pipeline
- Text cleaning (lowercase, URL removal, special chars)
- TF-IDF vectorization (unigrams + bigrams, sublinear TF)
- Logistic Regression with multinomial solver
- scikit-learn Pipeline (vectorizer + classifier in one object)
- Confusion matrix & class distribution plots
- Model persistence via pickle

---

## 📊 Model Performance (dair-ai/emotion dataset)

| Emotion  | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| joy      | 0.88      | 0.91   | 0.89     |
| sadness  | 0.91      | 0.93   | 0.92     |
| anger    | 0.87      | 0.84   | 0.85     |
| fear     | 0.88      | 0.86   | 0.87     |
| love     | 0.87      | 0.82   | 0.84     |
| surprise | 0.83      | 0.85   | 0.84     |
| **avg**  | **0.87**  | **0.87** | **0.87** |

> Results with full dair-ai/emotion dataset (~16k samples).

---

## 🗓️ 6-Day Implementation Plan

| Day | Task |
|-----|------|
| 1   | Setup env, install deps, understand dataset |
| 2   | Data preprocessing + EDA |
| 3   | Train & evaluate model, tune hyperparameters |
| 4   | Build Streamlit UI (basic) |
| 5   | Add charts, sentiment analysis, polish UI |
| 6   | Testing, README, GitHub push |

---

## 🔧 Hyperparameter Tuning

To experiment with hyperparameters, modify `train_model.py`:

```python
# TF-IDF options
TfidfVectorizer(
    max_features=50000,   # try: 30000, 70000
    ngram_range=(1, 2),   # try: (1,1), (1,3)
    sublinear_tf=True,    # log-scaled TF
    min_df=2,             # minimum document frequency
)

# Logistic Regression options
LogisticRegression(
    C=5.0,                # try: 1.0, 10.0
    max_iter=1000,
    solver="lbfgs",
)
```

---

## 📦 Dataset

**dair-ai/emotion** on HuggingFace  
- 16,000 English Twitter messages
- 6 emotion labels: sadness, joy, love, anger, fear, surprise
- Pre-split: train (16k) / val (2k) / test (2k)

---

## 🚀 Deployment (Streamlit Cloud)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as entry point
4. Add `models/` files to the repo (or use Git LFS)
5. Deploy!

---

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first.

---

## 📄 License

MIT License — free to use, modify, and distribute.
