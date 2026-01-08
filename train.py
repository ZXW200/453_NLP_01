"""
NLP预处理实验 - 模型训练与评估
SCC453 Natural Language Processing

输入：train_data.csv, test_data.csv
输出：实验结果对比
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 加载数据
print("加载数据...")
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
print(f"训练集: {len(train)}, 测试集: {len(test)}")

# 4种预处理方法
methods = ['baseline', 'no_stopwords', 'stemming', 'full']

# 2种模型
models = {
    'SVM': SVC(kernel='linear'),
    'LogisticRegression': LogisticRegression(max_iter=1000)
}

# 存储结果
results = []

print("\n=== 开始训练 ===")
for method in methods:
    # TF-IDF向量化
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train[method])
    X_test = tfidf.transform(test[method])
    y_train = train['label']
    y_test = test['label']

    for model_name, model in models.items():
        print(f"训练: {method} + {model_name}...")

        # 训练
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 评估
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'method': method,
            'model': model_name,
            'accuracy': acc,
            'f1': f1
        })
        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")

# 结果汇总
print("\n=== 结果汇总 ===")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 保存结果
results_df.to_csv('results.csv', index=False)
print("\n结果已保存: results.csv")

# 找最佳组合
best = results_df.loc[results_df['accuracy'].idxmax()]
print(f"\n最佳组合: {best['method']} + {best['model']}, Accuracy: {best['accuracy']:.4f}")