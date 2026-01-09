# plot.py
# 读取results.csv并生成图表

import os
import pandas as pd
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "data_out")
csv_path = os.path.join(data_dir, "results.csv")

order = ["baseline", "no_stopwords", "stemming", "full"]


def add_bar_labels(ax, fmt="{:.3f}"):
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, padding=3, fontsize=9)


def plot_bar(df, dataset, name, metric, ylabel):
    """通用柱状图"""
    data = df[df["Dataset"] == dataset]
    pivot = data.pivot(index="Preprocessing", columns="Model", values=metric).loc[order]

    ax = pivot.plot(kind="bar")
    ax.set_title(f"{dataset}: {ylabel} by Preprocessing")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel(ylabel)

    if metric == "Seconds":
        add_bar_labels(ax, fmt="{:.2f}")
    elif metric == "VocabSize":
        add_bar_labels(ax, fmt="{:.0f}")
    else:
        add_bar_labels(ax, fmt="{:.3f}")

    plt.tight_layout()
    path = os.path.join(data_dir, name)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def plot_delta(df, dataset, name):
    """MacroF1相对baseline的变化"""
    data = df[df["Dataset"] == dataset].copy()

    rows = []
    for model in data["Model"].unique():
        base_val = data[(data["Model"] == model) & (data["Preprocessing"] == "baseline")]["MacroF1"].iloc[0]
        for _, row in data[data["Model"] == model].iterrows():
            rows.append({
                "Preprocessing": row["Preprocessing"],
                "Model": model,
                "Delta": row["MacroF1"] - base_val
            })

    deltas = pd.DataFrame(rows)
    pivot = deltas.pivot(index="Preprocessing", columns="Model", values="Delta").loc[order]

    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{dataset}: ΔMacroF1 vs baseline")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel("ΔMacroF1")

    # 添加标签，baseline跳过（因为都是0会重叠）
    for i, c in enumerate(ax.containers):
        labels = [f"{v:+.3f}" if abs(v) > 0.0001 else "" for v in c.datavalues]
        ax.bar_label(c, labels=labels, padding=3, fontsize=9)

    plt.tight_layout()
    path = os.path.join(data_dir, name)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def plot_line(df, dataset, name):
    """MacroF1趋势线"""
    data = df[df["Dataset"] == dataset]

    plt.figure(figsize=(10, 6))
    x = list(range(len(order)))

    for model in data["Model"].unique():
        y = data[data["Model"] == model].set_index("Preprocessing").loc[order]["MacroF1"].tolist()
        plt.plot(x, y, marker="o", label=model, linewidth=2, markersize=8)
        for xi, yi in zip(x, y):
            plt.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, order)
    plt.title(f"{dataset}: MacroF1 Trend")
    plt.xlabel("Preprocessing")
    plt.ylabel("MacroF1")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(data_dir, name)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def plot_vocab_time(df, dataset, name):
    """词汇量和时间双Y轴折线图"""
    data = df[df["Dataset"] == dataset]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = list(range(len(order)))

    # 左Y轴 - 词汇量（只有一条线，因为两个模型用同一个TF-IDF）
    ax1.set_xlabel("Preprocessing")
    ax1.set_ylabel("Vocabulary Size", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # 取第一个模型的VocabSize（两个模型一样）
    vocab = data[data["Model"] == data["Model"].iloc[0]].set_index("Preprocessing").loc[order]["VocabSize"].tolist()
    ax1.plot(x, vocab, marker="o", linestyle="-", color="blue", linewidth=2, markersize=8, label="VocabSize")
    for xi, yi in zip(x, vocab):
        ax1.text(xi, yi + (max(vocab) - min(vocab)) * 0.05, f"{int(yi):,}",
                 ha="center", va="bottom", fontsize=10, color="blue", fontweight="bold")

    # 右Y轴 - 时间（两个模型分开画）
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time (seconds)", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")

    styles = {
        "LogisticRegression": {"color": "coral", "marker": "s", "label": "LR Time", "offset": 0.1},
        "LinearSVC": {"color": "darkred", "marker": "^", "label": "SVC Time", "offset": -0.1}
    }

    for model in data["Model"].unique():
        y = data[data["Model"] == model].set_index("Preprocessing").loc[order]["Seconds"].tolist()
        s = styles[model]
        ax2.plot(x, y, marker=s["marker"], linestyle="--", color=s["color"],
                 linewidth=2, markersize=8, label=s["label"])
        # 时间标签，两个模型错开一点
        for xi, yi in zip(x, y):
            ax2.text(xi + s["offset"], yi, f"{yi:.2f}s", ha="center", va="top",
                     fontsize=9, color=s["color"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(order)
    plt.title(f"{dataset}: Vocabulary Size & Training Time")

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    path = os.path.join(data_dir, name)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def main():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}. Run train.py first.")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from results.csv")

    for dataset in ["IMDb", "Emotion"]:
        if dataset not in df["Dataset"].unique():
            print(f"Warning: {dataset} not found, skipping.")
            continue

        print(f"\n=== {dataset} ===")

        # MacroF1图
        plot_delta(df, dataset, f"{dataset.lower()}_delta_bar.png")
        plot_line(df, dataset, f"{dataset.lower()}_line.png")

        # 词汇量+时间 双Y轴折线图
        plot_vocab_time(df, dataset, f"{dataset.lower()}_vocab_time.png")

    print("\nDone! All figures saved to:", data_dir)


if __name__ == "__main__":
    main()
