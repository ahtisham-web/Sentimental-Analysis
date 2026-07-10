"""
Full training pipeline covering every phase of the project spec:
EDA -> Cleaning -> Preprocessing (stem vs lemma) -> Features (BoW vs TF-IDF)
-> Train (LogReg, NB, SVM, RandomForest) -> Evaluate -> Save (.pkl)
"""
import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))
from utils import preprocess_text  # shared with backend -- no train/serve skew

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "sentiment.csv"
MODEL_PATH = PROJECT_ROOT / "backend" / "model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "backend" / "vectorizer.pkl"
MODEL_INFO_PATH = PROJECT_ROOT / "backend" / "model_info.json"
EDA_CHART_PATH = PROJECT_ROOT / "training" / "label_distribution.png"


# ---------- Phase 2: Dataset Exploration (EDA) ----------
def explore_dataset(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("PHASE 2: DATASET EXPLORATION")
    print("=" * 60)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nShape (rows, columns):", df.shape)
    print("\nInfo:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nClass distribution:")
    print(df["label"].value_counts())

    df["label"].value_counts().plot(kind="bar", color=["#ef4444", "#94a3b8", "#22c55e"])
    plt.title("Label Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_CHART_PATH)
    plt.close()
    print(f"\nSaved label distribution chart to {EDA_CHART_PATH}")


# ---------- Phase 4: Preprocessing comparison (stemming vs lemmatization) ----------
def compare_stem_vs_lemma(sample_texts: list) -> None:
    print("\n" + "=" * 60)
    print("PHASE 4: STEMMING vs LEMMATIZATION (sample comparison)")
    print("=" * 60)
    for t in sample_texts:
        stemmed = preprocess_text(t, method="stem")
        lemmatized = preprocess_text(t, method="lemmatize")
        print(f"\nOriginal:      {t}")
        print(f"Stemmed:       {stemmed}")
        print(f"Lemmatized:    {lemmatized}")


# ---------- Phase 5: Feature engineering comparison (BoW vs TF-IDF) ----------
def compare_bow_vs_tfidf(X_train, X_test, y_train, y_test) -> None:
    print("\n" + "=" * 60)
    print("PHASE 5: BAG-OF-WORDS vs TF-IDF (quick comparison, Logistic Regression)")
    print("=" * 60)

    bow = CountVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_bow = bow.fit_transform(X_train)
    X_test_bow = bow.transform(X_test)
    model_bow = LogisticRegression(max_iter=1000, C=5)
    model_bow.fit(X_train_bow, y_train)
    acc_bow = accuracy_score(y_test, model_bow.predict(X_test_bow))
    print(f"Bag-of-Words accuracy:  {acc_bow:.4f}")

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    model_tfidf = LogisticRegression(max_iter=1000, C=5)
    model_tfidf.fit(X_train_tfidf, y_train)
    acc_tfidf = accuracy_score(y_test, model_tfidf.predict(X_test_tfidf))
    print(f"TF-IDF accuracy:        {acc_tfidf:.4f}")
    print("\n-> TF-IDF is used for the production model (down-weights very common "
          "words across the corpus, typically generalizes slightly better than raw counts).")


def main():
    # ---------- Load + clean ----------
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])

    explore_dataset(df)

    compare_stem_vs_lemma([
        "I am not enjoying this at all, it's running badly.",
        "The players were happily running and jumping.",
    ])

    df["clean_text"] = df["text"].apply(lambda t: preprocess_text(t, method="lemmatize"))
    df = df[df["clean_text"].str.len() > 0]

    X = df["clean_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    compare_bow_vs_tfidf(X_train, X_test, y_train, y_test)

    # ---------- Phase 5 (production): TF-IDF ----------
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ---------- Phase 6: Model training (+ Random Forest) ----------
    print("\n" + "=" * 60)
    print("PHASE 6: MODEL TRAINING")
    print("=" * 60)
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=5),
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    best_name, best_model, best_acc = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        print(f"  {name}: accuracy = {acc:.4f}")
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")

    token_counts = df["clean_text"].apply(lambda t: len(t.split()))
    print("Average tokens per sentence after preprocessing:", round(float(np.mean(token_counts)), 2))

    # ---------- Phase 7: Evaluation ----------
    print("\n" + "=" * 60)
    print("PHASE 7: MODEL EVALUATION")
    print("=" * 60)
    y_pred = best_model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix (rows=true, cols=pred), labels order:", sorted(y.unique()))
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))

    # ---------- Phase 8: Save model ----------
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    model_info = {
        "model_name": best_name,
        "accuracy": round(best_acc * 100, 2),
        "precision": round(report_dict["weighted avg"]["precision"] * 100, 2),
        "recall": round(report_dict["weighted avg"]["recall"] * 100, 2),
        "f1_score": round(report_dict["weighted avg"]["f1-score"] * 100, 2),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model info to {MODEL_INFO_PATH}")

    # ---------- Sanity check ----------
    samples = [
        "I absolutely love this new update, great work!",
        "The package will arrive on Thursday.",
        "This is the worst experience I've ever had.",
    ]
    cleaned = [preprocess_text(s, method="lemmatize") for s in samples]
    vecs = vectorizer.transform(cleaned)
    preds = best_model.predict(vecs)
    print("\nSanity check predictions:")
    for s, p in zip(samples, preds):
        print(f"  '{s}' -> {p}")


if __name__ == "__main__":
    main()
