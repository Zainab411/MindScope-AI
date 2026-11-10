
import pickle
import re
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "mental_health.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

EXPECTED_CLASSES = [
    "Normal",
    "Anxiety",
    "Depression",
    "Suicidal"
]


def clean_text(text):
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


def normalize_label(label):
    value = str(label).strip().lower()

    labels = {
        "normal": "Normal",
        "anxiety": "Anxiety",
        "depression": "Depression",
        "suicidal": "Suicidal",
        "suicidal tendency": "Suicidal",
        "suicidal tendencies": "Suicidal",
        "suicidal_tendencies": "Suicidal"
    }

    return labels.get(value)


def main():

    print("\n===== MINDSCOPE AI TRAINING =====")

    if not DATA_PATH.exists():
        raise FileNotFoundError("mental_health.csv was not found.")

    df = pd.read_csv(DATA_PATH)

    if not {"text", "status"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain 'text' and 'status' columns."
        )

    print("\nOriginal dataset:")
    print(df["status"].value_counts(dropna=False))

    df = df[["text", "status"]].copy()
    df.dropna(subset=["text", "status"], inplace=True)

    df["text"] = df["text"].apply(clean_text)
    df["status"] = df["status"].apply(normalize_label)

    df = df[
        (df["text"] != "") &
        (df["status"].notna())
    ].copy()

    df.drop_duplicates(
        subset=["text", "status"],
        inplace=True
    )

    print("\nFour-category dataset:")
    print(df["status"].value_counts())

    available = set(df["status"].unique())
    missing = set(EXPECTED_CLASSES) - available

    if missing:
        raise ValueError(
            f"Missing categories: {sorted(missing)}"
        )

    if df["status"].value_counts().min() < 2:
        raise ValueError("Not enough examples in a category.")

    X = df["text"]
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n===== MODEL PERFORMANCE =====")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            predictions,
            labels=EXPECTED_CLASSES,
            zero_division=0
        )
    )

    print("\n===== CONFUSION MATRIX =====")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=EXPECTED_CLASSES
        )
    )

    # Save the final model using all available dataset records.
    final_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    all_text_vectorized = final_vectorizer.fit_transform(X)

    final_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    final_model.fit(all_text_vectorized, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(final_vectorizer, f)

    print("\n===== TRAINING COMPLETED =====")
    print("Categories:", list(final_model.classes_))
    print("Saved:", MODEL_PATH)
    print("Saved:", VECTORIZER_PATH)


if __name__ == "__main__":
    main()
