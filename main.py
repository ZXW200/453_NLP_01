"""
main.py - 数据预处理
下载IMDb和Emotion数据集，应用不同预处理策略
"""

import os
import re
import string
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

st = PorterStemmer()
lem = WordNetLemmatizer()

# 配置
seed = 42
test_ratio = 0.2
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_out")
os.makedirs(out_dir, exist_ok=True)

# 停用词表
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

# 否定词（情感分析要保留这些）
neg_words = {"no", "nor", "not", "never", "but", "however"}
smart_stops = stopwords - neg_words

html_re = re.compile(r"<.*?>")
url_re = re.compile(r'http\S+|www\S+')
num_re = re.compile(r'\b\d+\b')
punct_tbl = str.maketrans("", "", string.punctuation)


# -------- 预处理函数 --------

def baseline(text):
    """只去HTML和URL，保留其他所有内容"""
    text = str(text)
    text = re.sub(html_re, " ", text)
    text = re.sub(url_re, " ", text)
    return text.lower().strip()


def no_stopwords(text):
    """去HTML + 去标点 + 去停用词"""
    text = str(text)
    text = re.sub(html_re, " ", text)
    text = re.sub(url_re, " ", text)
    text = text.translate(punct_tbl)
    text = text.lower()
    words = [w for w in text.split() if w not in stopwords and len(w) > 1]
    return " ".join(words)


def stemming(text):
    """Porter词干提取 + 去数字 + 去停用词"""
    text = str(text)
    text = re.sub(html_re, " ", text)
    text = re.sub(url_re, " ", text)
    text = re.sub(num_re, " ", text)
    text = text.translate(punct_tbl)
    text = text.lower()
    words = [st.stem(w) for w in text.split() if w not in stopwords and len(w) > 1]
    return " ".join(words)


def full(text):
    """Lemmatization + 保留否定词"""
    text = str(text)
    text = re.sub(html_re, " ", text)
    text = re.sub(url_re, " ", text)
    text = text.translate(punct_tbl)
    text = text.lower()
    words = [lem.lemmatize(w) for w in text.split() if w not in smart_stops and len(w) > 1]
    return " ".join(words)


funcs = {
    "baseline": baseline,
    "no_stopwords": no_stopwords,
    "stemming": stemming,
    "full": full,
}


def do_preprocess(df):
    """对text列应用所有预处理方法"""
    df = df.copy()
    for name, fn in funcs.items():
        df[name] = df["text"].apply(fn)
    return df


def main():
    print("Output dir:", out_dir)

    # -------- 1) IMDb数据集 --------
    print("\n下载IMDb数据集...")
    imdb_dir = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

    # 找到CSV文件
    imdb_path = os.path.join(imdb_dir, "IMDB Dataset.csv")
    if not os.path.exists(imdb_path):
        # 如果不在根目录，找一下子目录
        for root, dirs, files in os.walk(imdb_dir):
            for f in files:
                if f.endswith(".csv"):
                    imdb_path = os.path.join(root, f)
                    break

    print(f"读取: {imdb_path}")
    imdb = pd.read_csv(imdb_path)

    # 处理列名和标签
    imdb = imdb.rename(columns={"review": "text", "sentiment": "label"})
    imdb["label"] = imdb["label"].map({"positive": 1, "negative": 0})
    print(f"IMDb: {len(imdb)} 条, 标签分布: {imdb['label'].value_counts().to_dict()}")

    # 划分训练测试集
    imdb_train, imdb_test = train_test_split(
        imdb, test_size=test_ratio, random_state=seed, stratify=imdb["label"]
    )
    imdb_train = do_preprocess(imdb_train.reset_index(drop=True))
    imdb_test = do_preprocess(imdb_test.reset_index(drop=True))

    # 保存
    cols = ["text", "label", "baseline", "no_stopwords", "stemming", "full"]
    imdb_train[cols].to_csv(os.path.join(out_dir, "imdb_train.csv"), index=False)
    imdb_test[cols].to_csv(os.path.join(out_dir, "imdb_test.csv"), index=False)
    print(f"保存: imdb_train.csv ({len(imdb_train)}条), imdb_test.csv ({len(imdb_test)}条)")

    # -------- 2) Emotion数据集 --------
    print("\n下载Emotion数据集...")
    emo_dir = kagglehub.dataset_download("nelgiriyewithana/emotions")

    # 找到CSV文件
    emo_path = os.path.join(emo_dir, "text.csv")
    if not os.path.exists(emo_path):
        for root, dirs, files in os.walk(emo_dir):
            for f in files:
                if f.endswith(".csv"):
                    emo_path = os.path.join(root, f)
                    break

    print(f"读取: {emo_path}")
    emo = pd.read_csv(emo_path)

    # 处理标签（字符串转数字）
    if emo["label"].dtype == object:
        emo_list = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        emo["label"] = emo["label"].str.lower().map({n: i for i, n in enumerate(emo_list)})
    print(f"Emotion: {len(emo)} 条, 标签分布: {emo['label'].value_counts().to_dict()}")

    # 划分训练测试集
    emo_train, emo_test = train_test_split(
        emo, test_size=test_ratio, random_state=seed, stratify=emo["label"]
    )
    emo_train = do_preprocess(emo_train.reset_index(drop=True))
    emo_test = do_preprocess(emo_test.reset_index(drop=True))

    # 保存
    emo_train[cols].to_csv(os.path.join(out_dir, "emotion_train.csv"), index=False)
    emo_test[cols].to_csv(os.path.join(out_dir, "emotion_test.csv"), index=False)
    print(f"保存: emotion_train.csv ({len(emo_train)}条), emotion_test.csv ({len(emo_test)}条)")

    print("\n完成！可以运行 train.py 了")


if __name__ == "__main__":
    main()
