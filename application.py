"""
Multi-Class Text Emotion Detection System
Streamlit Web Application
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EMOTION_EMOJI = {
    "joy":      "😄",
    "sadness":  "😢",
    "anger":    "😠",
    "fear":     "😨",
    "love":     "❤️",
    "surprise": "😲",
}

EMOTION_COLOR = {
    "joy":      "#FFD700",
    "sadness":  "#4169E1",
    "anger":    "#DC143C",
    "fear":     "#8B008B",
    "love":     "#FF69B4",
    "surprise": "#00CED1",
}

EMOTION_SENTIMENT = {
    "joy":      "positive",
    "love":     "positive",
    "surprise": "neutral",
    "sadness":  "negative",
    "fear":     "negative",
    "anger":    "negative",
}

# ─────────────────────────────────────────────
# TEXT UTILS
# ─────────────────────────────────────────────

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sentiment_from_text(text):
    """
    Rule-based sentiment analysis using positive/negative word lists.
    Lightweight – no extra library needed.
    """
    positive_words = {
        "good", "great", "excellent", "wonderful", "amazing", "fantastic",
        "happy", "love", "joy", "best", "beautiful", "awesome", "nice",
        "brilliant", "perfect", "delightful", "cheerful", "glad", "thrilled",
        "pleased", "fortunate", "blessed", "grateful", "excited", "superb",
    }
    negative_words = {
        "bad", "terrible", "awful", "horrible", "hate", "sad", "angry",
        "fear", "worst", "ugly", "disgusting", "dreadful", "miserable",
        "depressed", "worried", "anxious", "pain", "hurt", "fail", "poor",
        "broken", "scared", "furious", "awful", "nightmare", "stress",
    }
    words = set(preprocess_text(text).split())
    pos = len(words & positive_words)
    neg = len(words & negative_words)
    if pos > neg:
        return "positive", pos, neg
    elif neg > pos:
        return "negative", pos, neg
    else:
        return "neutral", pos, neg


def confidence_label(prob):
    if prob >= 0.70:
        return "🟢 High", "green"
    elif prob >= 0.45:
        return "🟡 Medium", "orange"
    else:
        return "🔴 Low", "red"


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    model_path = "models/emotion_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(
            "⚠️ Model not found! Please run `python train_model.py` first, "
            "then restart the app."
        )
        st.stop()
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict(pipeline, text):
    clean = preprocess_text(text)
    probs = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    top_idx = np.argmax(probs)
    return classes[top_idx], probs, classes


# ─────────────────────────────────────────────
# PLOTLY BAR CHART
# ─────────────────────────────────────────────

def probability_chart(classes, probs, predicted):
    colors = [EMOTION_COLOR.get(c, "#888") for c in classes]
    fig = go.Figure(go.Bar(
        x=[f"{EMOTION_EMOJI.get(c,'')} {c.capitalize()}" for c in classes],
        y=[round(p * 100, 2) for p in probs],
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Emotion",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 110]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        height=380,
        margin=dict(t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Emotion Detector",
        page_icon="🧠",
        layout="centered",
    )

    # ── Header ──────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 5px 0">
            <h1 style="font-size:2.4rem; margin-bottom:4px">🧠 EmotiSense AI</h1>
             <!-- <p style="color:#888; font-size:1rem">
                 Multi-Class Text Emotion Detection · TF-IDF + Logistic Regression
             </p> -->
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    pipeline = load_model()

    # ── Input ────────────────────────────────
    st.subheader("📝 Enter Your Text")
    user_text = st.text_area(
        label="Type or paste text below:",
        placeholder="e.g. I am so excited about this new project!",
        height=130,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze = st.button("Analyze", type="primary", use_container_width=True)
    with col2:
        st.write("")

    # ── Examples ─────────────────────────────
    # with st.expander("💡 Try example sentences"):
    #     examples = {
    #         "Joy":      "I just got amazing news and I am absolutely thrilled!",
    #         "Sadness":  "I feel so lonely and empty, nothing seems right anymore.",
    #         "Anger":    "This is completely unacceptable, I am furious right now!",
    #         "Fear":     "I am terrified about what might happen next.",
    #         "Love":     "You mean everything to me, I love you so much.",
    #         "Surprise": "I can't believe this happened, I am totally shocked!",
    #     }
    #     for emotion, sentence in examples.items():
    #         if st.button(f"{EMOTION_EMOJI[emotion.lower()]} {emotion}: \"{sentence[:55]}...\"",
    #                      key=f"ex_{emotion}"):
    #             st.session_state["example_text"] = sentence
    #             st.rerun()

    # # Use example if selected
    # if "example_text" in st.session_state and not user_text.strip():
    #     user_text = st.session_state.pop("example_text")

    # st.divider()

    # ── Analysis ─────────────────────────────
    if analyze and user_text.strip():

        emotion, probs, classes = predict(pipeline, user_text)
        top_prob = max(probs)
        conf_label, conf_color = confidence_label(top_prob)
        sentiment, pos_count, neg_count = get_sentiment_from_text(user_text)
        sentiment_icon = {"positive": "😊", "negative": "😟", "neutral": "😐"}[sentiment]
        emotion_sentiment = EMOTION_SENTIMENT.get(emotion, "neutral")
        match = sentiment == emotion_sentiment

        # ── Main result card ────────────────
        emoji = EMOTION_EMOJI.get(emotion, "🤔")
        color = EMOTION_COLOR.get(emotion, "#888")
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}11);
                border: 2px solid {color};
                border-radius: 16px;
                padding: 24px 28px;
                text-align: center;
                margin-bottom: 20px;
            ">
                <div style="font-size: 4rem; line-height:1.1">{emoji}</div>
                <div style="font-size: 2rem; font-weight: 700; color: {color};
                            text-transform: capitalize; margin: 6px 0">{emotion}</div>
                <div style="font-size: 1.05rem; color: #555;">
                    Confidence: <b style="color:{conf_color}">{conf_label}</b>
                    &nbsp;|&nbsp; Top Probability: <b>{top_prob*100:.1f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Probability chart ────────────────
        fig = probability_chart(classes, probs, emotion)
        st.plotly_chart(fig, use_container_width=True)

        # ── Probability table ─────────────────
        with st.expander("📊 Full probability table"):
            prob_df = pd.DataFrame({
                "Emotion": [f"{EMOTION_EMOJI.get(c,'')} {c.capitalize()}" for c in classes],
                "Probability": [f"{p*100:.2f}%" for p in probs],
            }).sort_values("Probability", ascending=False).reset_index(drop=True)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # ── Sentiment Analysis ─────────────────
        st.subheader("💬 Sentiment Analysis")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            st.metric("Text Sentiment", f"{sentiment_icon} {sentiment.capitalize()}")
        with s_col2:
            st.metric("Emotion Sentiment", f"{EMOTION_SENTIMENT.get(emotion,'neutral').capitalize()}")
        with s_col3:
            st.metric("Agreement", "✅ Match" if match else "⚠️ Mismatch")

        if match:
            st.success(f"Sentiment and emotion both indicate a **{sentiment}** signal.")
        else:
            st.warning(
                f"Text sentiment is **{sentiment}** but detected emotion "
                f"(**{emotion}**) suggests a **{emotion_sentiment}** tone. "
                "This can occur with complex or mixed emotional language."
            )

        # ── Input analysis ────────────────────
        st.caption(f"📝 Input: {len(user_text.split())} words · "
                   f"Positive indicators: {pos_count} · Negative indicators: {neg_count}")

    elif analyze:
        st.warning("⚠️ Please enter some text before clicking Analyze.")

    # ── Sidebar ──────────────────────────────
    # with st.sidebar:
    #     st.header("ℹ️ About")
    #     st.markdown(
    #         """
    #         **Emotions detected:**

    #         | Emoji | Emotion  |
    #         |-------|----------|
    #         | 😄    | Joy      |
    #         | 😢    | Sadness  |
    #         | 😠    | Anger    |
    #         | 😨    | Fear     |
    #         | ❤️    | Love     |
    #         | 😲    | Surprise |

    #         **Model stack:**
    #         - 🔤 TF-IDF (unigrams + bigrams)
    #         - 📈 Logistic Regression
    #         - 📦 scikit-learn Pipeline

    #         **Dataset:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
    #         """
    #     )
    #     st.divider()
    #     st.header("🔍 Confidence Guide")
    #     st.markdown(
    #         """
    #         - 🟢 **High** ≥ 70%
    #         - 🟡 **Medium** 45–70%
    #         - 🔴 **Low** < 45%
    #         """
    #     )
    #     st.divider()
    #     st.caption("Built with Streamlit · scikit-learn · Plotly")


if __name__ == "__main__":
    main()
