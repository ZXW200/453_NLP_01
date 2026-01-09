# plot_results.py
# ------------------------------------
# Read data_out/results.csv and generate:
#   - IMDb bar (MacroF1) + value labels
#   - IMDb delta bar (ΔMacroF1 vs baseline) + value labels
#   - IMDb line (MacroF1) + point labels
#   - Emotion bar (MacroF1) + value labels
#   - Emotion delta bar (ΔMacroF1 vs baseline) + value labels
#   - Emotion line (MacroF1) + point labels

import os
import pandas as pd
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, "data_out")
csv_path = os.path.join(data_dir, "results.csv")

order = ["baseline", "no_stopwords", "stemming", "full"]
metric = "MacroF1"


def add_bar_labels(ax, fmt="{:.3f}"):
    """Add numeric labels on bar tops."""
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, padding=3, fontsize=9)


def plot_bar(df, dataset, name):
    """Absolute bar chart with value labels."""
    data = df[df["Dataset"] == dataset]
    pivot = data.pivot(index="Preprocessing", columns="Model", values=metric).loc[order]

    ax = pivot.plot(kind="bar")
    ax.set_title(f"{dataset}: {metric} by Preprocessing")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel(metric)
    add_bar_labels(ax, fmt="{:.3f}")

    plt.tight_layout()
    path = os.path.join(data_dir, name)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def plot_delta(df, dataset, name):
    """Delta bar chart (relative to baseline) with +/- value labels."""
    data = df[df["Dataset"] == dataset].copy()

    rows = []
    for model in data["Model"].unique():
        base = data[(data["Model"] == model) & (data["Preprocessing"] == "baseline")][metric].iloc[0]
        for _, row in data[data["Model"] == model].iterrows():
            rows.append({
                "Preprocessing": row["Preprocessing"],
                "Model": model,
                "Delta": row[metric] - base
            })

    deltas = pd.DataFrame(rows)
    pivot = deltas.pivot(index="Preprocessing", columns="Model", values="Delta").loc[order]

    ax = pivot.plot(kind="bar")
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{dataset}: Δ{metric} vs baseline")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel(f"Δ{metric}")
    add_bar_labels(ax, fmt="{:+.3f}")

    plt.tight_layout()
    path = os.path.join(data_dir, name)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def plot_line(df, dataset, name):
    """Line chart with point labels."""
    data = df[df["Dataset"] == dataset]

    plt.figure()
    x = list(range(len(order)))

    for model in data["Model"].unique():
        y = data[data["Model"] == model].set_index("Preprocessing").loc[order][metric].tolist()
        plt.plot(x, y, marker="o", label=model)

        for xi, yi in zip(x, y):
            plt.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, order)
    plt.title(f"{dataset}: {metric} Trend")
    plt.xlabel("Preprocessing")
    plt.ylabel(metric)
    plt.legend()

    plt.tight_layout()
    path = os.path.join(data_dir, name)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def main():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}. Run train.py first.")

    df = pd.read_csv(csv_path)

    need = {"Dataset", "Preprocessing", "Model", metric}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"results.csv missing columns: {miss}. Found: {df.columns.tolist()}")

    for dataset in ["IMDb", "Emotion"]:
        if dataset not in set(df["Dataset"].unique()):
            print(f"Warning: dataset '{dataset}' not found in results.csv. Skipping.")
            continue

        plot_bar(df, dataset, f"{dataset.lower()}_bar.png")
        plot_delta(df, dataset, f"{dataset.lower()}_delta_bar.png")
        plot_line(df, dataset, f"{dataset.lower()}_line.png")

    print("\nDone. All figures are in:", data_dir)


if __name__ == "__main__":
    main()
