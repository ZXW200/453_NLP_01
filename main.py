"""
NLP预处理实验 - 数据加载与预处理
SCC453 Natural Language Processing

输出：
  - train_data.csv
  - test_data.csv

列：
  text, label, baseline, no_stopwords, stemming, full
"""

import os
import pandas as pd
import re
import string

# -------------------------
# Stopwords（保留否定词）
# -------------------------
STOP_WORDS = {
    'i', 'me', 'my', 'we', 'you', 'he', 'she', 'it', 'they', 'the', 'a', 'an',
    'and', 'but', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'this', 'that', 'these', 'those', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'as', 'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'can', 'now',

    # ⚠️ 注意：不把 no/nor/not 加进来
}
NEGATIONS = {"no", "nor", "not", "never"}
STOP_WORDS = STOP_WORDS - NEGATIONS

HTML_RE = re.compile(r"<.*?>")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ============ 数据集 ============
def load_imdb_data(sample_size=5000, seed=42):
    """
    从Kaggle下载并加载IMDB数据集
    KaggleHub需要：pip install kagglehub
    """
    import kagglehub

    print("正在从Kaggle下载数据集...")
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    print(f"下载完成! 文件路径: {path}")

    csv_path = os.path.join(path, "IMDB Dataset.csv")

    # 优先 utf-8，失败再 latin-1

    df = pd.read_csv(csv_path, encoding="utf-8")


    print(f"加载完成! 共 {len(df)} 条数据")

    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df["label"] = df["label"].map({"positive": 1, "negative": 0})

    # 采样（可复现）
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)

    # shuffle（非常关键）
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ============ 预处理函数 ============
def preprocess_baseline(text: str) -> str:
    """基线：移除HTML + 小写"""
    text = re.sub(HTML_RE, " ", text)
    return text.lower()

def preprocess_no_stopwords(text: str) -> str:
    """基线 + 去停用词（保留否定词）"""
    text = preprocess_baseline(text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

def preprocess_stemming(text: str) -> str:
    """基线 + 简单stemming（演示用，论文里要说明是简化规则）"""
    text = preprocess_baseline(text)
    words = text.split()
    stemmed = []
    for w in words:
        if w.endswith("ing") and len(w) > 5:
            w = w[:-3]
        elif w.endswith("ed") and len(w) > 4:
            w = w[:-2]
        elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            w = w[:-1]
        stemmed.append(w)
    return " ".join(stemmed)

def preprocess_full(text: str) -> str:
    """基线 + 去标点 + 去停用词 + stemming"""
    text = preprocess_baseline(text)
    text = text.translate(PUNCT_TABLE)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]

    stemmed = []
    for w in words:
        if w.endswith("ing") and len(w) > 5:
            w = w[:-3]
        elif w.endswith("ed") and len(w) > 4:
            w = w[:-2]
        stemmed.append(w)
    return " ".join(stemmed)


# ============ 主程序 ============
if __name__ == "__main__":
    SEED = 42
    SAMPLE_SIZE = 5000

    print("加载IMDB数据集...")
    df = load_imdb_data(sample_size=SAMPLE_SIZE, seed=SEED)
    print(f"样本数: {len(df)}")
    print(f"正面: {(df['label']==1).sum()}, 负面: {(df['label']==0).sum()}")

    # 划分训练/测试集（shuffle已在 load_imdb_data 里做了）
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

    preprocess_funcs = {
        "baseline": preprocess_baseline,
        "no_stopwords": preprocess_no_stopwords,
        "stemming": preprocess_stemming,
        "full": preprocess_full,
    }

    # 对比预处理效果
    print("\n=== 预处理对比 ===")
    sample = train_df["text"].iloc[0]
    print(f"原文: {sample[:120]}...")
    for name, func in preprocess_funcs.items():
        print(f"{name}: {func(sample)[:120]}...")

    # 应用预处理（Series.map更快更简洁）
    print("\n=== 应用预处理 ===")
    for name, func in preprocess_funcs.items():
        train_df[name] = train_df["text"].map(func)
        test_df[name] = test_df["text"].map(func)

        orig_words = train_df["text"].map(lambda t: len(str(t).split())).sum()
        proc_words = train_df[name].map(lambda t: len(str(t).split())).sum()
        reduction = (orig_words - proc_words) / max(orig_words, 1) * 100
        print(f"{name}: 词数减少 {reduction:.1f}%")

    # 保存（保留原始text列用于错误分析）
    train_df[["text", "label", "baseline", "no_stopwords", "stemming", "full"]].to_csv(
        "train_data.csv", index=False
    )
    test_df[["text", "label", "baseline", "no_stopwords", "stemming", "full"]].to_csv(
        "test_data.csv", index=False
    )

    print("\n数据已保存: train_data.csv, test_data.csv")
