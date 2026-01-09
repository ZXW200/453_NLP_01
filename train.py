"""
train.py
------------------------------------
Train & evaluate models on:
  - IMDb (binary)
  - Emotion (6-class)

Input CSVs (from ./data_out):
  imdb_train.csv, imdb_test.csv
  emotion_train.csv, emotion_test.csv

Outputs:
  results.csv (in ./data_out)
"""

import os
import time
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_out")

PREPROCESS_COLS = ["baseline", "no_stopwords", "stemming", "full"]

TFIDF_KWARGS = dict(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

MODELS = {
    # n_jobs removed to avoid sklearn warning
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    "LinearSVC": LinearSVC(dual="auto", max_iter=2000, random_state=42),
}

def run_dataset(dataset_name: str, train_path: str, test_path: str) -> pd.DataFrame:
    train_df = pd.read_csv(train_path).fillna("")
    test_df = pd.read_csv(test_path).fillna("")

    y_train = train_df["label"].astype(int).values
    y_test = test_df["label"].astype(int).values

    rows = []

    for col in PREPROCESS_COLS:
        vec = TfidfVectorizer(**TFIDF_KWARGS)
        X_train = vec.fit_transform(train_df[col].astype(str))
        X_test = vec.transform(test_df[col].astype(str))
        vocab_size = len(vec.vocabulary_)

        for model_name, model in MODELS.items():
            t0 = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            sec = time.time() - t0

            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")  # works for binary + multiclass

            rows.append({
                "Dataset": dataset_name,
                "Preprocessing": col,
                "Model": model_name,
                "VocabSize": vocab_size,
                "Accuracy": acc,
                "MacroF1": macro_f1,
                "Seconds": sec
            })

            print(f"[{dataset_name}] {col:12s} + {model_name:16s} "
                  f"Acc={acc:.4f} MacroF1={macro_f1:.4f} V={vocab_size} ({sec:.2f}s)")

    return pd.DataFrame(rows)

def main():
    imdb_train = os.path.join(DATA_DIR, "imdb_train.csv")
    imdb_test = os.path.join(DATA_DIR, "imdb_test.csv")
    emo_train = os.path.join(DATA_DIR, "emotion_train.csv")
    emo_test = os.path.join(DATA_DIR, "emotion_test.csv")

    for p in [imdb_train, imdb_test, emo_train, emo_test]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}. Run main.py first.")

    res = []
    res.append(run_dataset("IMDb", imdb_train, imdb_test))
    res.append(run_dataset("Emotion", emo_train, emo_test))

    res_df = pd.concat(res, ignore_index=True)
    out_path = os.path.join(DATA_DIR, "results.csv")
    res_df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("\nTop rows by MacroF1:")
    print(res_df.sort_values(["Dataset", "MacroF1"], ascending=[True, False]).head(12).to_string(index=False))

if __name__ == "__main__":
    main()
