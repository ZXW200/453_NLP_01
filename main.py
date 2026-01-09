"""
main.py
------------------------------------
Load (via kagglehub):
  1) IMDb 50k movie reviews (binary sentiment)
  2) Emotions dataset (6-class emotion classification)

Preprocess strategies:
  baseline, no_stopwords, stemming, full

Outputs (in ./data_out):
  imdb_train.csv, imdb_test.csv
  emotion_train.csv, emotion_test.csv

Each CSV columns:
  text, label, baseline, no_stopwords, stemming, full
"""

import os
import re
import string
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# 第一次运行需要下载词形还原词典（只需一次）
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
# -------------------------
# Config
# -------------------------
SEED = 42
TEST_SIZE = 0.2
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_out")
os.makedirs(OUT_DIR, exist_ok=True)

IMDB_KAGGLE_ID = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
EMOTION_KAGGLE_ID = "nelgiriyewithana/emotions"  # 6 emotions dataset on Kaggle

# -------------------------
# 基础停用词表
BASIC_STOP_WORDS = {
    'i', 'me', 'my', 'we', 'you', 'he', 'she', 'it', 'they', 'the', 'a', 'an',
    'and', 'but', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'this', 'that', 'these', 'those', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'as', 'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
}

# 否定词和转折词（情感分析的关键！）
NEGATION_WORDS = {"no", "nor", "not", "never", "but", "however"}

# 激进组/正常组使用的停用词（全部删掉）
ALL_STOP_WORDS = BASIC_STOP_WORDS

# 全面组使用的停用词（保留否定词，防止语义反转）
SMART_STOP_WORDS = BASIC_STOP_WORDS - NEGATION_WORDS

HTML_RE = re.compile(r"<.*?>")
URL_RE = re.compile(r'http\S+|www\S+')
NUMBER_RE = re.compile(r'\b\d+\b')
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# -------------------------
# Preprocessing functions (Modified)
# -------------------------

def preprocess_baseline(text: str) -> str:
    """
    [对照组] Raw / Tokenization Only
    策略：只去HTML和URL，保留绝大多数信息（包括停用词、标点、数字）。
    目的：测试'不做处理'的底线效果。
    """
    text = str(text)
    text = re.sub(HTML_RE, " ", text)  # 必须去HTML，否则是乱码
    text = re.sub(URL_RE, " ", text)
    text = text.lower().strip()  # 统一小写，否则词表爆炸
    # 不去标点，不去停用词，不去数字
    return text


def preprocess_no_stopwords(text: str) -> str:
    """
    [正常组] Standard / Bag-of-Words
    策略：去HTML + 去标点 + 去除 *所有* 停用词。
    目的：工业界最通用的清洗方式，减少特征维度。
    缺点：会丢失 'not good' 中的 'not'。
    """
    text = str(text)
    text = re.sub(HTML_RE, " ", text)
    text = re.sub(URL_RE, " ", text)
    text = text.translate(PUNCT_TABLE)  # 去标点
    text = text.lower()

    words = text.split()
    # 过滤所有停用词
    words = [w for w in words if w not in ALL_STOP_WORDS and len(w) > 1]
    return " ".join(words)


def preprocess_stemming(text: str) -> str:
    """
    [激进组] Aggressive / Stemming
    策略：PorterStemmer (砍词尾) + 去数字 + 去所有停用词。
    目的：最大程度降维，将 'acting', 'acted' 强行归一为 'act'。
    缺点：丢失时态、语态和词的精细含义，丢失否定词。
    """
    text = str(text)
    text = re.sub(HTML_RE, " ", text)
    text = re.sub(URL_RE, " ", text)
    text = re.sub(NUMBER_RE, " ", text)  # 激进去除数字
    text = text.translate(PUNCT_TABLE)
    text = text.lower()

    words = text.split()
    # 强力词干提取
    stemmed = [stemmer.stem(w) for w in words if w not in ALL_STOP_WORDS and len(w) > 1]
    return " ".join(stemmed)


def preprocess_full(text: str) -> str:
    """
    [全面组] Comprehensive / Context-Aware
    策略：Lemmatization (还原原型) + *保留* 否定词(not) + 保留数字意义(可选)。
    目的：在降维的同时，保留 'not bad' 和 'bad' 的区别。
    """
    text = str(text)
    text = re.sub(HTML_RE, " ", text)
    text = re.sub(URL_RE, " ", text)
    text = text.translate(PUNCT_TABLE)
    text = text.lower()

    words = text.split()
    # 1. 使用 SMART_STOP_WORDS (保留了 not, no, but)
    # 2. 使用 Lemmatizer (比 Stemming 温和，better -> good)
    cleaned = []
    for w in words:
        if w not in SMART_STOP_WORDS and len(w) > 1:
            cleaned.append(lemmatizer.lemmatize(w))

    return " ".join(cleaned)


PREPROCESS_FUNCS = {
    "baseline": preprocess_baseline,  # 1. 不处理 (Raw)
    "no_stopwords": preprocess_no_stopwords,  # 2. 正常 (Standard)
    "stemming": preprocess_stemming,  # 3. 激进 (Aggressive)
    "full": preprocess_full,  # 4. 全面 (Comprehensive)
}
# -------------------------
# Kaggle helpers
# -------------------------
def find_csv_files(root_dir: str):
    csvs = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return csvs

def read_best_csv(download_dir: str) -> pd.DataFrame:
    """
    Find CSVs under download_dir, prefer the largest one (usually main data).
    """
    csvs = find_csv_files(download_dir)
    if not csvs:
        raise FileNotFoundError(f"No CSV found under: {download_dir}")

    # sort by file size (desc)
    csvs.sort(key=lambda p: os.path.getsize(p), reverse=True)
    csv_path = csvs[0]
    print(f"Using CSV: {csv_path} ({os.path.getsize(csv_path)/1024/1024:.2f} MB)")

    # robust encoding
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")

def normalise_text_label_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Make sure df has columns: text, label (int).
    Handles common variants.
    """
    df = df.copy()

    # Identify text column
    text_candidates = ["text", "review", "sentence", "content", "tweet"]
    text_col = None
    for c in text_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"[{dataset_name}] Cannot find a text column. Columns={df.columns.tolist()}")

    # Identify label column
    label_candidates = ["label", "sentiment", "emotion", "category", "target"]
    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"[{dataset_name}] Cannot find a label column. Columns={df.columns.tolist()}")

    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].fillna("").astype(str)

    # Map labels
    if dataset_name.lower().startswith("imdb"):
        # IMDb sentiment likely string -> map
        if df["label"].dtype == object:
            df["label"] = df["label"].map({"positive": 1, "negative": 0})
        df["label"] = df["label"].astype(int)
    else:
        # Emotion dataset may already be int 0..5 or string emotions
        if df["label"].dtype == object:
            # common 6 emotions
            order = ["sadness", "joy", "love", "anger", "fear", "surprise"]
            mapping = {name: i for i, name in enumerate(order)}
            df["label"] = df["label"].str.lower().map(mapping)
        df["label"] = df["label"].astype(int)

    return df[["text", "label"]]

def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, fn in PREPROCESS_FUNCS.items():
        df[name] = df["text"].map(fn)
    return df

def save_split(train_df: pd.DataFrame, test_df: pd.DataFrame, prefix: str):
    cols = ["text", "label", "baseline", "no_stopwords", "stemming", "full"]
    train_path = os.path.join(OUT_DIR, f"{prefix}_train.csv")
    test_path = os.path.join(OUT_DIR, f"{prefix}_test.csv")
    train_df[cols].to_csv(train_path, index=False)
    test_df[cols].to_csv(test_path, index=False)
    print(f"Saved: {train_path} (rows={len(train_df)})")
    print(f"Saved: {test_path} (rows={len(test_df)})")

def main():
    print("Output dir:", OUT_DIR)

    # -------------------------
    # 1) IMDb
    # -------------------------
    print("\nDownloading IMDb via kagglehub...")
    imdb_dir = kagglehub.dataset_download(IMDB_KAGGLE_ID)
    print("IMDb download dir:", imdb_dir)

    imdb_raw = read_best_csv(imdb_dir)
    imdb = normalise_text_label_columns(imdb_raw, "imdb")
    print("IMDb rows:", len(imdb), "label counts:", imdb["label"].value_counts().to_dict())

    imdb_train, imdb_test = train_test_split(
        imdb, test_size=TEST_SIZE, random_state=SEED, stratify=imdb["label"]
    )
    imdb_train = apply_preprocessing(imdb_train.reset_index(drop=True))
    imdb_test = apply_preprocessing(imdb_test.reset_index(drop=True))
    save_split(imdb_train, imdb_test, "imdb")

    # -------------------------
    # 2) Emotion dataset (6-class)
    # -------------------------
    print("\nDownloading Emotion dataset via kagglehub...")
    emo_dir = kagglehub.dataset_download(EMOTION_KAGGLE_ID)
    print("Emotion download dir:", emo_dir)

    emo_raw = read_best_csv(emo_dir)
    emo = normalise_text_label_columns(emo_raw, "emotion")
    print("Emotion rows:", len(emo), "label counts:", emo["label"].value_counts().to_dict())

    emo_train, emo_test = train_test_split(
        emo, test_size=TEST_SIZE, random_state=SEED, stratify=emo["label"]
    )
    emo_train = apply_preprocessing(emo_train.reset_index(drop=True))
    emo_test = apply_preprocessing(emo_test.reset_index(drop=True))
    save_split(emo_train, emo_test, "emotion")

    print("\nDone. You can now run train.py")

if __name__ == "__main__":
    main()
