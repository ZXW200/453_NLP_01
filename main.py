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

st = PorterStemmer()
lem = WordNetLemmatizer()
# -------------------------
# Config
# -------------------------
seed = 42
test_ratio = 0.2
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_out")
os.makedirs(out_dir, exist_ok=True)

imdb_id = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
emo_id = "nelgiriyewithana/emotions"  # 6 emotions dataset on Kaggle

# -------------------------
# 基础停用词表
stopwords = {
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
neg_words = {"no", "nor", "not", "never", "but", "however"}

# 激进组/正常组使用的停用词（全部删掉）
all_stops = stopwords

# 全面组使用的停用词（保留否定词，防止语义反转）
smart_stops = stopwords - neg_words

html_re = re.compile(r"<.*?>")
url_re = re.compile(r'http\S+|www\S+')
num_re = re.compile(r'\b\d+\b')
punct_tbl = str.maketrans("", "", string.punctuation)


# -------------------------
# Preprocessing functions (Modified)
# -------------------------

def clean1(text: str) -> str:
    """
    [对照组] Raw / Tokenization Only
    策略：只去HTML和URL，保留绝大多数信息（包括停用词、标点、数字）。
    目的：测试'不做处理'的底线效果。
    """
    text = str(text)
    text = re.sub(html_re, " ", text)  # 必须去HTML，否则是乱码
    text = re.sub(url_re, " ", text)
    text = text.lower().strip()  # 统一小写，否则词表爆炸
    # 不去标点，不去停用词，不去数字
    return text


def clean2(text: str) -> str:
    """
    [正常组] Standard / Bag-of-Words
    策略：去HTML + 去标点 + 去除 *所有* 停用词。
    目的：工业界最通用的清洗方式，减少特征维度。
    缺点：会丢失 'not good' 中的 'not'。
    """
    text = str(text)
    text = re.sub(html_re, " ", text)
    text = re.sub(url_re, " ", text)
    text = text.translate(punct_tbl)  # 去标点
    text = text.lower()

    words = text.split()
    # 过滤所有停用词
    words = [w for w in words if w not in all_stops and len(w) > 1]
    return " ".join(words)


def clean3(text: str) -> str:
    """
    [激进组] Aggressive / Stemming
    策略：PorterStemmer (砍词尾) + 去数字 + 去所有停用词。
    目的：最大程度降维，将 'acting', 'acted' 强行归一为 'act'。
    缺点：丢失时态、语态和词的精细含义，丢失否定词。
    """
    text = str(text)
    text = re.sub(html_re, " ", text)
    text = re.sub(url_re, " ", text)
    text = re.sub(num_re, " ", text)  # 激进去除数字
    text = text.translate(punct_tbl)
    text = text.lower()

    words = text.split()
    # 强力词干提取
    res = [st.stem(w) for w in words if w not in all_stops and len(w) > 1]
    return " ".join(res)


def clean4(text: str) -> str:
    """
    [全面组] Comprehensive / Context-Aware
    策略：Lemmatization (还原原型) + *保留* 否定词(not) + 保留数字意义(可选)。
    目的：在降维的同时，保留 'not bad' 和 'bad' 的区别。
    """
    text = str(text)
    text = re.sub(html_re, " ", text)
    text = re.sub(url_re, " ", text)
    text = text.translate(punct_tbl)
    text = text.lower()

    words = text.split()
    # 1. 使用 smart_stops (保留了 not, no, but)
    # 2. 使用 Lemmatizer (比 Stemming 温和，better -> good)
    res = []
    for w in words:
        if w not in smart_stops and len(w) > 1:
            res.append(lem.lemmatize(w))

    return " ".join(res)


funcs = {
    "baseline": clean1,  # 1. 不处理 (Raw)
    "no_stopwords": clean2,  # 2. 正常 (Standard)
    "stemming": clean3,  # 3. 激进 (Aggressive)
    "full": clean4,  # 4. 全面 (Comprehensive)
}
# -------------------------
# Kaggle helpers
# -------------------------
def get_csvs(dir: str):
    lst = []
    for root, _, files in os.walk(dir):
        for f in files:
            if f.lower().endswith(".csv"):
                lst.append(os.path.join(root, f))
    return lst

def load_csv(dir: str) -> pd.DataFrame:
    """
    Find CSVs under dir, prefer the largest one (usually main data).
    """
    lst = get_csvs(dir)
    if not lst:
        raise FileNotFoundError(f"No CSV found under: {dir}")

    # sort by file size (desc)
    lst.sort(key=lambda p: os.path.getsize(p), reverse=True)
    path = lst[0]
    print(f"Using CSV: {path} ({os.path.getsize(path)/1024/1024:.2f} MB)")

    # robust encoding
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def fix_cols(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Make sure df has columns: text, label (int).
    Handles common variants.
    """
    df = df.copy()

    # Identify text column
    txt_cols = ["text", "review", "sentence", "content", "tweet"]
    txt = None
    for c in txt_cols:
        if c in df.columns:
            txt = c
            break
    if txt is None:
        raise ValueError(f"[{name}] Cannot find a text column. Columns={df.columns.tolist()}")

    # Identify label column
    lab_cols = ["label", "sentiment", "emotion", "category", "target"]
    lab = None
    for c in lab_cols:
        if c in df.columns:
            lab = c
            break
    if lab is None:
        raise ValueError(f"[{name}] Cannot find a label column. Columns={df.columns.tolist()}")

    df = df.rename(columns={txt: "text", lab: "label"})
    df["text"] = df["text"].fillna("").astype(str)

    # Map labels
    if name.lower().startswith("imdb"):
        # IMDb sentiment likely string -> map
        if df["label"].dtype == object:
            df["label"] = df["label"].map({"positive": 1, "negative": 0})
        df["label"] = df["label"].astype(int)
    else:
        # Emotion dataset may already be int 0..5 or string emotions
        if df["label"].dtype == object:
            # common 6 emotions
            emo_list = ["sadness", "joy", "love", "anger", "fear", "surprise"]
            mp = {n: i for i, n in enumerate(emo_list)}
            df["label"] = df["label"].str.lower().map(mp)
        df["label"] = df["label"].astype(int)

    return df[["text", "label"]]

def do_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, fn in funcs.items():
        df[name] = df["text"].map(fn)
    return df

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, prefix: str):
    cols = ["text", "label", "baseline", "no_stopwords", "stemming", "full"]
    p1 = os.path.join(out_dir, f"{prefix}_train.csv")
    p2 = os.path.join(out_dir, f"{prefix}_test.csv")
    train_df[cols].to_csv(p1, index=False)
    test_df[cols].to_csv(p2, index=False)
    print(f"Saved: {p1} (rows={len(train_df)})")
    print(f"Saved: {p2} (rows={len(test_df)})")

def main():
    print("Output dir:", out_dir)

    # -------------------------
    # 1) IMDb
    # -------------------------
    print("\nDownloading IMDb via kagglehub...")
    dir1 = kagglehub.dataset_download(imdb_id)
    print("IMDb download dir:", dir1)

    raw1 = load_csv(dir1)
    data1 = fix_cols(raw1, "imdb")
    print("IMDb rows:", len(data1), "label counts:", data1["label"].value_counts().to_dict())

    train1, test1 = train_test_split(
        data1, test_size=test_ratio, random_state=seed, stratify=data1["label"]
    )
    train1 = do_preprocess(train1.reset_index(drop=True))
    test1 = do_preprocess(test1.reset_index(drop=True))
    save_data(train1, test1, "imdb")

    # -------------------------
    # 2) Emotion dataset (6-class)
    # -------------------------
    print("\nDownloading Emotion dataset via kagglehub...")
    dir2 = kagglehub.dataset_download(emo_id)
    print("Emotion download dir:", dir2)

    raw2 = load_csv(dir2)
    data2 = fix_cols(raw2, "emotion")
    print("Emotion rows:", len(data2), "label counts:", data2["label"].value_counts().to_dict())

    train2, test2 = train_test_split(
        data2, test_size=test_ratio, random_state=seed, stratify=data2["label"]
    )
    train2 = do_preprocess(train2.reset_index(drop=True))
    test2 = do_preprocess(test2.reset_index(drop=True))
    save_data(train2, test2, "emotion")

    print("\nDone. You can now run train.py")

if __name__ == "__main__":
    main()
