
from flask import Flask, render_template, request, jsonify
import pickle
import re
import random
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# =====================================================
# ALLOWED CATEGORIES
# =====================================================

ALLOWED_CLASSES = {
    "Normal",
    "Anxiety",
    "Depression",
    "Suicidal"
}

if set(model.classes_) != ALLOWED_CLASSES:
    raise ValueError(
        "The loaded model must contain exactly: "
        "Normal, Anxiety, Depression and Suicidal. "
        "Run train_model.py again."
    )


# =====================================================
# TEXT CLEANING
# Must match train_model.py
# =====================================================

def clean(text):

    text = str(text).lower().strip()

    replacements = {
        "i'm": "i am",
        "im": "i am",
        "i've": "i have",
        "i'll": "i will",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
        "couldn't": "could not",
        "won't": "will not",
        "isn't": "is not",
        "it's": "it is",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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
        "🌸 You do not have to handle everything alone.",
        "🫂 Be kind to yourself. Small steps are important too."
    ],

    "Suicidal": [
        "💙 You matter. Please reach out to someone you trust.",
        "🫂 You do not have to face difficult feelings alone.",
        "💗 Please consider contacting a trusted person or mental health professional."
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
# HOME ROUTE
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# PREDICTION ROUTE
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True) or {}

        user_text = data.get("text", "").strip()

        if not user_text:
            return jsonify({
                "error": "Please enter some text."
            }), 400

        cleaned_text = clean(user_text)

        if not cleaned_text:
            return jsonify({
                "error": "Please enter meaningful text."
            }), 400

        transformed_text = vectorizer.transform([cleaned_text])

        probabilities = model.predict_proba(transformed_text)[0]

        classes = list(model.classes_)

        predicted_index = probabilities.argmax()

        emotion = classes[predicted_index]

        confidence = float(
            probabilities[predicted_index] * 100
        )

        distribution = {
            category: round(float(probability * 100), 2)
            for category, probability in zip(classes, probabilities)
        }

        severity = get_severity(emotion)

        message = random.choice(messages[emotion])

        return jsonify({
            "result": emotion,
            "confidence": round(confidence, 2),
            "severity": severity,
            "message": message,
            "distribution": distribution,
            "ml_result": emotion
        })

    except Exception as e:

        app.logger.exception("Prediction error")

        return jsonify({
            "error": "Unable to analyze the text. Please try again."
        }), 500


# =====================================================
# RUN FLASK
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
