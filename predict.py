"""
Quick CLI / notebook-friendly inference utility.
Usage:
    python predict.py "I am so happy today!"
    python predict.py  # interactive mode
"""

import pickle
import numpy as np
import re
import sys
import os

EMOTION_EMOJI = {
    "joy": "😄", "sadness": "😢", "anger": "😠",
    "fear": "😨", "love": "❤️", "surprise": "😲",
}

EMOTION_SENTIMENT = {
    "joy": "positive", "love": "positive", "surprise": "neutral",
    "sadness": "negative", "fear": "negative", "anger": "negative",
}


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pipeline(model_dir="models"):
    path = os.path.join(model_dir, "emotion_pipeline.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python train_model.py` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(pipeline, text):
    clean = preprocess_text(text)
    probs  = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    top_idx = np.argmax(probs)
    emotion = classes[top_idx]
    return emotion, probs, classes


def confidence_label(p):
    if p >= 0.70: return "High"
    if p >= 0.45: return "Medium"
    return "Low"


def display_result(text, pipeline):
    emotion, probs, classes = predict(pipeline, text)
    top_prob = max(probs)
    emoji = EMOTION_EMOJI.get(emotion, "?")

    print("\n" + "="*50)
    print(f"Input : {text}")
    print(f"Emotion : {emoji}  {emotion.upper()}")
    print(f"Confidence : {top_prob*100:.1f}%  ({confidence_label(top_prob)})")
    print(f"Sentiment (emotion-based): {EMOTION_SENTIMENT.get(emotion,'neutral')}")
    print("\n── Probability Distribution ──")
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        bar = "█" * int(probs[i] * 30)
        print(f"  {classes[i]:<9} {probs[i]*100:5.1f}%  {bar}")
    print("="*50)


if __name__ == "__main__":
    pipeline = load_pipeline()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        display_result(text, pipeline)
    else:
        print("🧠 Emotion Detector – Interactive Mode (type 'quit' to exit)")
        while True:
            text = input("\nEnter text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                display_result(text, pipeline)
