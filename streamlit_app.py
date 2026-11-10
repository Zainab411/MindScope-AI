
import streamlit as st
import pickle
import re
import random
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MindScope AI",
    page_icon="🧠",
    layout="centered"
)


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =====================================================
# EMOTION LABELS
# =====================================================

def normalize_emotion(emotion):
    value = str(emotion).strip().lower()

    labels = {
        "normal": "Normal",
        "anxiety": "Anxiety",
        "depression": "Depression",
        "suicidal": "Suicidal",
        "suicidal tendency": "Suicidal",
        "suicidal tendencies": "Suicidal",
        "suicidal_tendencies": "Suicidal"
    }

    return labels.get(value, str(emotion).title())


# =====================================================
# SUPPORTIVE MESSAGES
# =====================================================

messages = {
    "Normal": [
        "😊 You seem to be doing well. Keep taking care of yourself! 💙",
        "🌸 Keep maintaining a healthy balance in your life!",
        "✨ Remember to make time for yourself."
    ],
    "Anxiety": [
        "🫶 You are not alone. Take things one step at a time.",
        "🌷 Try some slow breathing and give yourself a moment.",
        "💙 Your feelings matter. Be gentle with yourself."
    ],
    "Depression": [
        "💗 You matter, and your feelings deserve care and attention.",
        "🌸 You don't have to handle everything alone.",
        "🫂 Be kind to yourself. Small steps are important too."
    ],
    "Suicidal": [
        "💙 You matter. Please reach out to someone you trust.",
        "🫂 You do not have to face difficult feelings alone.",
        "💗 Please consider contacting a trusted person or mental health professional now."
    ]
}


# =====================================================
# SEVERITY
# =====================================================

def get_severity(emotion):
    if emotion == "Suicidal":
        return "Critical 🔴"
    elif emotion == "Depression":
        return "High 🟠"
    elif emotion == "Anxiety":
        return "Moderate 🟡"
    return "Low 🟢"


# =====================================================
# CHOCOLATE THEME CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #F5E8D8, #EAD2BA);
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* MAIN HEADING */

.main-heading {
    color: #4A2618;
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
}

/* SUBTITLE */

.subtitle {
    color: #704633;
    font-size: 16px;
    text-align: center;
    margin-bottom: 28px;
}

/* SECTION HEADING */

.section-heading {
    color: #4A2618;
    font-size: 25px;
    font-weight: 750;
    margin-top: 24px;
    margin-bottom: 15px;
}

/* GENERAL TEXT */

p, label {
    color: #4A3024;
}

/* TEXT AREA */

textarea {
    background: #FFFCF8 !important;
    color: #40251B !important;
    border: 2px solid #9A7058 !important;
    border-radius: 13px !important;
    font-size: 15px !important;
}

/* ANALYZE BUTTON */

.stButton > button {
    background: linear-gradient(90deg, #70402E, #9A6676) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 11px 25px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

.stButton > button:hover {
    background: #603625 !important;
    color: white !important;
}

/* RESULT CARD */

.result-card {
    background: #4A2B20;
    border: 1px solid #684735;
    border-radius: 18px;
    padding: 22px;
    margin-top: 15px;
    margin-bottom: 15px;
    text-align: center;
}

.result-card h2 {
    color: #FFF8F0 !important;
    font-size: 25px;
    font-weight: 750;
}

.result-card p {
    color: #FFF8F0 !important;
    font-size: 17px;
    font-weight: 650;
    margin-bottom: 12px;
}

/* SUPPORTIVE MESSAGE CARD */

.message-card {
    background: #FFFCF8;
    border-left: 5px solid #9A6676;
    border-radius: 15px;
    padding: 18px;
    margin-top: 15px;
    margin-bottom: 15px;
}

.message-card h3 {
    color: #5A3425 !important;
    font-size: 20px;
    margin-bottom: 10px;
}

.message-card p {
    color: #4A3024 !important;
    font-size: 15px;
}

/* HISTORY CARD */

.history-card {
    background: #FFFCF8;
    border: 1px solid #E1C9B5;
    padding: 13px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: #4A3024;
    font-size: 14px;
}

.history-card p {
    color: #4A3024 !important;
}

/* DIVIDER */

hr {
    border-color: #D5BBA5 !important;
}

/* MOBILE RESPONSIVENESS */

@media screen and (max-width: 600px) {
    .main-heading {
        font-size: 29px;
    }

    .section-heading {
        font-size: 21px;
    }

    .result-card h2 {
        font-size: 21px;
    }

    .result-card p {
        font-size: 15px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# =====================================================
# MAIN HEADING
# =====================================================

st.markdown(
    '<div class="main-heading">🧠 MindScope AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">💙 Your feelings matter. Let AI understand you.</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# USER INPUT
# =====================================================

st.markdown(
    '<div class="section-heading">🌷 How Are You Feeling Today?</div>',
    unsafe_allow_html=True
)

user_text = st.text_area(
    "Share your thoughts and feelings:",
    placeholder="Write about how you are feeling...",
    height=120
)

analyze_button = st.button(
    "Analyze 🚀",
    use_container_width=True
)


# =====================================================
# PREDICTION
# =====================================================

if analyze_button:

    if not user_text.strip():
        st.warning("⚠️ Please enter some text before analyzing.")

    else:
        cleaned_text = clean_text(user_text)

        if not cleaned_text:
            st.warning("⚠️ Please enter meaningful text.")

        else:
            transformed_text = vectorizer.transform([cleaned_text])

            prediction = model.predict(transformed_text)[0]

            probabilities = model.predict_proba(transformed_text)[0]

            classes = list(model.classes_)

            predicted_index = classes.index(prediction)

            confidence = float(probabilities[predicted_index] * 100)

            distribution = {}

            for i, class_name in enumerate(classes):
                display_name = normalize_emotion(class_name)
                distribution[display_name] = float(probabilities[i] * 100)

            emotion = normalize_emotion(prediction)

            severity = get_severity(emotion)

            supportive_message = random.choice(
                messages.get(
                    emotion,
                    ["💙 Your feelings matter. Take care of yourself."]
                )
            )

            result = {
                "emotion": emotion,
                "confidence": confidence,
                "severity": severity,
                "message": supportive_message,
                "distribution": distribution
            }

            st.session_state.latest_result = result

            st.session_state.history.insert(0, result)

            st.session_state.history = st.session_state.history[:5]


# =====================================================
# DISPLAY EMOTIONAL ANALYSIS
# =====================================================

if st.session_state.latest_result is not None:

    result = st.session_state.latest_result

    emotion = result["emotion"]
    confidence = result["confidence"]
    severity = result["severity"]
    supportive_message = result["message"]
    distribution = result["distribution"]

    st.divider()

    st.markdown(
        '<div class="section-heading">🌷 Your Emotional Analysis</div>',
        unsafe_allow_html=True
    )

    # RESULT CARD

    st.markdown(f"""
<div class="result-card">
<h2>📋 Detected Emotion: {emotion}</h2>
<p>📊 ML Confidence: {confidence:.1f}%</p>
<p>⚡ Severity: {severity}</p>
</div>
""", unsafe_allow_html=True)

    # SUPPORTIVE MESSAGE

    st.markdown(f"""
<div class="message-card">
<h3>💌 A Little Message for You</h3>
<p>{supportive_message}</p>
</div>
""", unsafe_allow_html=True)

    # SAFETY MESSAGE

    if emotion == "Suicidal":
        st.warning(
            "💙 This prediction is not a diagnosis. If you may be in "
            "immediate danger or thinking about harming yourself, "
            "please contact emergency services or a trusted person now. "
            "In India, call 112 for emergency assistance."
        )

    # CONFIDENCE GRAPH

    st.markdown(
        '<div class="section-heading">📊 Emotion Confidence Overview</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(3.8, 1.7))

    labels = list(distribution.keys())
    values = list(distribution.values())

    bars = ax.bar(labels, values, width=0.5)

    ax.set_ylabel("Confidence (%)", fontsize=8)

    ax.set_title(
        "ML Prediction Probabilities",
        fontsize=9,
        fontweight="bold"
    )

    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.tick_params(axis="y", labelsize=7)

    ax.set_ylim(0, 105)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            fontsize=6
        )

    fig.tight_layout()

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.pyplot(fig, use_container_width=False)

    plt.close(fig)


# =====================================================
# RECENT ANALYSIS HISTORY
# =====================================================

if st.session_state.history:

    st.divider()

    st.markdown(
        '<div class="section-heading">🕒 Recent Analysis History</div>',
        unsafe_allow_html=True
    )

    for index, item in enumerate(st.session_state.history):

        st.markdown(f"""
<div class="history-card">
<b>Analysis {index + 1}</b>
<p>Detected Emotion: {item["emotion"]}</p>
<p>Confidence: {item["confidence"]:.1f}%</p>
<p>Severity: {item["severity"]}</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div style="text-align:center; color:#5B321E; font-size:13px; margin-top:20px;">
💜 Made with care | MindScope AI
</div>
""", unsafe_allow_html=True)
