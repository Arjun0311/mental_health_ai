import pickle

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Text Analyzer",
    page_icon="🧠",
    layout="centered",
)

EMOTION_META = {
    "depression": {"emoji": "😔", "color": "#6C7A96", "tone": "warning"},
    "stress":     {"emoji": "😣", "color": "#D98E48", "tone": "warning"},
    "anxiety":    {"emoji": "😟", "color": "#C9A34E", "tone": "warning"},
    "anger":      {"emoji": "😠", "color": "#C4553D", "tone": "warning"},
    "happy":      {"emoji": "😊", "color": "#4C9A6A", "tone": "success"},
    "neutral":    {"emoji": "😐", "color": "#7A8B99", "tone": "info"},
}
DEFAULT_META = {"emoji": "🙂", "color": "#7A8B99", "tone": "info"}

CUSTOM_CSS = """
<style>
.app-header {
    text-align: center;
    padding-bottom: 0.5rem;
}
.app-header h1 {
    margin-bottom: 0.1rem;
    font-size: 2.1rem;
}
.app-subtitle {
    color: #8A94A6;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}
.result-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    color: white;
}
.result-card .emoji {
    font-size: 2.4rem;
    line-height: 1;
}
.result-card .label {
    font-size: 1.35rem;
    font-weight: 600;
    text-transform: capitalize;
    margin-top: 0.3rem;
}
.result-card .confidence {
    font-size: 0.9rem;
    opacity: 0.9;
    margin-top: 0.2rem;
}
.disclaimer-box {
    background-color: rgba(150, 150, 150, 0.08);
    border-left: 4px solid #8A94A6;
    padding: 0.7rem 1rem;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #8A94A6;
    margin-top: 1.5rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer


model, vectorizer = load_artifacts()
CLASS_LABELS = list(getattr(model, "classes_", []))

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🧠 Emotion Text Analyzer</h1>
        <div class="app-subtitle">
            A lightweight demo that classifies the emotional tone of a short piece of text.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.write(
        "This is a demo text-emotion classifier (TF-IDF + logistic regression) "
        "trained on a small synthetic dataset. It's built for exploring how a "
        "text-classification pipeline works end to end."
    )
    st.markdown("### Detected categories")
    for emo, meta in EMOTION_META.items():
        st.markdown(f"{meta['emoji']} &nbsp; {emo.capitalize()}", unsafe_allow_html=True)
    st.divider()
    st.caption(
        "⚠️ This tool is trained on synthetic example data. It is **not** a "
        "diagnostic or clinical instrument, and its predictions can be wrong "
        "— especially on phrasing it hasn't seen before."
    )

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
examples = [
    "Choose an example…",
    "I feel hopeless and tired lately, nothing seems to help.",
    "I'm so stressed about deadlines I can barely think straight.",
    "Today was calm and easy, nothing much happened.",
    "I got the job offer and I'm absolutely thrilled!",
]
example_choice = st.selectbox("Try an example, or write your own below:", examples)

default_text = "" if example_choice == examples[0] else example_choice
user_input = st.text_area(
    "How are you feeling today?",
    value=default_text,
    height=120,
    placeholder="Type a sentence or two about how you're feeling...",
)

analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------------
if analyze_clicked:
    if not user_input.strip():
        st.error("Please enter some text first.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]

        probabilities = None
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed)[0]
            confidence = max(probabilities)

        meta = EMOTION_META.get(prediction, DEFAULT_META)

        confidence_line = f"{confidence:.0%} confidence" if confidence is not None else ""
        st.markdown(
            f"""
            <div class="result-card" style="background-color: {meta['color']};">
                <div class="emoji">{meta['emoji']}</div>
                <div class="label">{prediction}</div>
                <div class="confidence">{confidence_line}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if prediction in ["depression", "stress", "anxiety"]:
            st.warning("You might be experiencing some emotional distress. Consider talking to someone you trust, or a mental health professional.")
        elif prediction == "happy":
            st.success("Great to hear you're feeling positive!")
        else:
            st.info("Thanks for sharing how you're feeling.")

        if probabilities is not None and CLASS_LABELS:
            with st.expander("See full breakdown across all categories"):
                breakdown = pd.DataFrame(
                    {"probability": probabilities}, index=CLASS_LABELS
                ).sort_values("probability", ascending=False)
                st.bar_chart(breakdown)

# ----------------------------------------------------------------------------
# Footer disclaimer
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-box">
        This app is a technical demo, not a mental health service. If you're
        going through a hard time, please reach out to a trusted person, a
        licensed professional, or a local crisis line — this tool is not a
        substitute for real support.
    </div>
    """,
    unsafe_allow_html=True,
)
