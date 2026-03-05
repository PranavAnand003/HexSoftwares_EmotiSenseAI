"""
Multi-Class Text Emotion Detection System
Training Script
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(data_dir="data"):
    """
    Load the Emotions dataset.
    Expected files: train.txt, val.txt, test.txt  (from dair-ai/emotion on HuggingFace)
    Each line: "text;label"
    Falls back to a small synthetic dataset if files not found.
    """
    def read_file(path):
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ";" in line:
                    text, label = line.rsplit(";", 1)
                    texts.append(text.strip())
                    labels.append(label.strip())
        return texts, labels

    train_path = os.path.join(data_dir, "train.txt")
    val_path   = os.path.join(data_dir, "val.txt")
    test_path  = os.path.join(data_dir, "test.txt")

    if os.path.exists(train_path):
        print("✅ Loading dataset from files...")
        tr_t, tr_l = read_file(train_path)
        va_t, va_l = read_file(val_path)
        te_t, te_l = read_file(test_path)

        all_texts  = tr_t + va_t + te_t
        all_labels = tr_l + va_l + te_l
        df = pd.DataFrame({"text": all_texts, "label": all_labels})
    else:
        print("⚠️  Dataset files not found. Generating synthetic data...")
        df = generate_synthetic_data()

    print(f"📊 Total samples: {len(df)}")
    print(df["label"].value_counts())
    return df


def generate_synthetic_data():
    """Small synthetic dataset for demo/testing purposes."""
    data = {
        "joy": [
            "I am so happy today!", "This is the best day of my life!",
            "I feel wonderful and excited!", "Life is beautiful and full of joy",
            "I just got promoted, feeling amazing!", "Today was absolutely fantastic",
            "I love spending time with my family", "Everything is going perfectly well",
            "I feel on top of the world right now", "What a delightful surprise!",
            "I am thrilled about the good news", "Pure happiness fills my heart",
            "I am overjoyed with this result", "Feeling blessed and grateful today",
            "This made me smile so much", "Best feeling in the world!",
            "I couldn't be happier right now", "Such a joyful moment",
            "My heart is full of happiness", "Celebrating this wonderful achievement",
        ],
        "sadness": [
            "I am feeling very sad and lonely", "This is the worst day ever",
            "I can't stop crying, everything hurts", "I feel so hopeless and depressed",
            "Lost my best friend, heartbroken", "Nothing seems to be going right",
            "I feel empty and worthless", "Tears won't stop falling down my face",
            "Missing someone who is no longer here", "Everything feels so dark right now",
            "I have no motivation to do anything", "The pain is unbearable today",
            "I feel abandoned by everyone", "So much grief in my heart",
            "I don't see the point anymore", "Feeling completely broken inside",
            "My heart aches so deeply", "Overcome with sorrow and despair",
            "Life feels meaningless without them", "I've never felt so alone",
        ],
        "anger": [
            "I am absolutely furious right now!", "This makes me so angry!",
            "How dare they treat me like this", "I am livid about what happened",
            "This is completely unacceptable behavior", "I want to scream I am so mad",
            "They have no right to do this to me", "I hate when people lie to my face",
            "Burning with rage at this injustice", "Stop testing my patience!",
            "I am outraged by this decision", "This is beyond infuriating",
            "My blood is boiling right now", "How could they be so selfish",
            "I am fed up with this nonsense", "Sick and tired of being disrespected",
            "This makes me so irritated", "I can barely control my anger",
            "Furious doesn't even begin to describe it", "They've crossed the line",
        ],
        "fear": [
            "I am terrified of what might happen", "Something feels very wrong here",
            "I can't shake this feeling of dread", "I am scared and don't know why",
            "The darkness makes me so afraid", "I fear the worst is coming",
            "My heart races with anxiety and fear", "I don't feel safe at all",
            "Something is lurking in the shadows", "I am paralyzed with fear",
            "The thought of it terrifies me", "I am horrified by what I saw",
            "Nightmares keep waking me up", "I feel a deep sense of foreboding",
            "Panic is taking over my mind", "I dread what tomorrow might bring",
            "My hands tremble with fear", "I'm frightened and can't calm down",
            "The anxiety is overwhelming", "I am afraid I might lose everything",
        ],
        "love": [
            "I love you more than words can say", "You are my everything",
            "My heart beats only for you", "I adore every little thing about you",
            "You make my world complete", "Falling deeper in love every day",
            "I cherish every moment with you", "You are the love of my life",
            "I feel so warm and loved around you", "My love for you knows no bounds",
            "You complete me in every way", "I am deeply in love with you",
            "Every moment with you is magical", "You are my soulmate and best friend",
            "I treasure our love so much", "Loving you feels so natural",
            "You make my heart sing with joy", "I am so grateful to have you",
            "My love for you grows stronger every day", "You are my greatest blessing",
        ],
        "surprise": [
            "I can't believe what just happened!", "This is totally unexpected!",
            "Wow, I didn't see that coming at all", "I am completely shocked right now",
            "What a stunning turn of events!", "I am astonished beyond words",
            "This is absolutely mind-blowing!", "I never expected this in a million years",
            "Speechless at this incredible news", "Completely taken by surprise",
            "I am stunned by this revelation", "No way, this can't be real!",
            "I am blown away by this outcome", "Jaw dropping moment right here",
            "Utterly shocked at what I witnessed", "This is beyond my wildest imagination",
            "I am flabbergasted by the news", "What a remarkable surprise this is",
            "I am amazed at this unexpected twist", "Could not have predicted this ever",
        ],
    }

    rows = []
    for label, texts in data.items():
        for text in texts:
            rows.append({"text": text, "label": label})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_text(text):
    """Clean and normalize text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)               # remove mentions/hashtags
    text = re.sub(r"[^a-zA-Z\s']", " ", text)           # keep letters + apostrophes
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    return text


# ─────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────

def train(df):
    df["clean_text"] = df["text"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📦 Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Pipeline: TF-IDF → Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )),
    ])

    print("\n🚀 Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc * 100:.2f}%")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    return pipeline, X_test, y_test, y_pred, acc


# ─────────────────────────────────────────────
# 4. EVALUATION PLOTS
# ─────────────────────────────────────────────

def save_confusion_matrix(y_test, y_pred, labels, output_path="assets/confusion_matrix.png"):
    os.makedirs("assets", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix – Emotion Detection", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved → {output_path}")


def save_class_distribution(df, output_path="assets/class_distribution.png"):
    os.makedirs("assets", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#FFD700", "#4169E1", "#DC143C", "#8B008B", "#FF69B4", "#00CED1"]
    df["label"].value_counts().plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.set_title("Emotion Class Distribution", fontsize=14)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📊 Class distribution saved → {output_path}")


# ─────────────────────────────────────────────
# 5. SAVE MODEL
# ─────────────────────────────────────────────

def save_model(pipeline, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    # Save full pipeline
    with open(os.path.join(model_dir, "emotion_pipeline.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

    # Save components separately too (for flexibility)
    with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(pipeline.named_steps["tfidf"], f)

    with open(os.path.join(model_dir, "lr_classifier.pkl"), "wb") as f:
        pickle.dump(pipeline.named_steps["clf"], f)

    with open(os.path.join(model_dir, "label_classes.pkl"), "wb") as f:
        pickle.dump(pipeline.classes_, f)

    print(f"💾 Model saved → {model_dir}/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data("data")

    save_class_distribution(df)

    pipeline, X_test, y_test, y_pred, acc = train(df)

    labels = sorted(df["label"].unique().tolist())
    save_confusion_matrix(y_test, y_pred, labels)

    save_model(pipeline)

    print(f"\n🎉 Training complete! Final Accuracy: {acc * 100:.2f}%")
    if acc >= 0.80:
        print("✅ Target accuracy (≥80%) achieved!")
    else:
        print("⚠️  Accuracy below 80%. Consider using the full HuggingFace dataset.")
