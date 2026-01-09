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

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "data_out")

cols = ["baseline", "no_stopwords", "stemming", "full"]

tfidf_args = dict(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

models = {
    # n_jobs removed to avoid sklearn warning
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    "LinearSVC": LinearSVC(dual="auto", max_iter=2000, random_state=42),
}

def run(name: str, train_path: str, test_path: str):
    df1 = pd.read_csv(train_path).fillna("")
    df2 = pd.read_csv(test_path).fillna("")

    y1 = df1["label"].astype(int).values
    y2 = df2["label"].astype(int).values

    results = []

    for c in cols:
        vec = TfidfVectorizer(**tfidf_args)
        x1 = vec.fit_transform(df1[c].astype(str))
        x2 = vec.transform(df2[c].astype(str))
        v_size = len(vec.vocabulary_)

        for m_name, m in models.items():
            t0 = time.time()
            m.fit(x1, y1)
            pred = m.predict(x2)
            t = time.time() - t0

            acc = accuracy_score(y2, pred)
            f1 = f1_score(y2, pred, average="macro")  # works for binary + multiclass

            results.append({
                "Dataset": name,
                "Preprocessing": c,
                "Model": m_name,
                "VocabSize": v_size,
                "Accuracy": acc,
                "MacroF1": f1,
                "Seconds": t
            })

            print(f"[{name}] {c:12s} + {m_name:16s} "
                  f"Acc={acc:.4f} MacroF1={f1:.4f} V={v_size} ({t:.2f}s)")

    return pd.DataFrame(results)

def main():
    p1 = os.path.join(data_dir, "imdb_train.csv")
    p2 = os.path.join(data_dir, "imdb_test.csv")
    p3 = os.path.join(data_dir, "emotion_train.csv")
    p4 = os.path.join(data_dir, "emotion_test.csv")

    for p in [p1, p2, p3, p4]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}. Run main.py first.")

    res = []
    res.append(run("IMDb", p1, p2))
    res.append(run("Emotion", p3, p4))

    df = pd.concat(res, ignore_index=True)
    out = os.path.join(data_dir, "results.csv")
    df.to_csv(out, index=False)

    print("\nSaved:", out)
    print("\nTop rows by MacroF1:")
    print(df.sort_values(["Dataset", "MacroF1"], ascending=[True, False]).head(12).to_string(index=False))

if __name__ == "__main__":
    main()

