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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_out")
CSV_PATH = os.path.join(DATA_DIR, "results.csv")

ORDER = ["baseline", "no_stopwords", "stemming", "full"]
METRIC = "MacroF1"


def _add_bar_labels(ax, fmt="{:.3f}"):
    """Add numeric labels on bar tops."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=9)


def plot_bar(df, dataset, out_name):
    """Absolute bar chart with value labels."""
    sub = df[df["Dataset"] == dataset]
    pivot = sub.pivot(index="Preprocessing", columns="Model", values=METRIC).loc[ORDER]

    ax = pivot.plot(kind="bar")
    ax.set_title(f"{dataset}: {METRIC} by Preprocessing")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel(METRIC)
    _add_bar_labels(ax, fmt="{:.3f}")

    plt.tight_layout()
    out = os.path.join(DATA_DIR, out_name)
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def plot_delta_bar(df, dataset, out_name):
    """Delta bar chart (relative to baseline) with +/- value labels."""
    sub = df[df["Dataset"] == dataset].copy()

    rows = []
    for model in sub["Model"].unique():
        base = sub[(sub["Model"] == model) & (sub["Preprocessing"] == "baseline")][METRIC].iloc[0]
        for _, r in sub[sub["Model"] == model].iterrows():
            rows.append({
                "Preprocessing": r["Preprocessing"],
                "Model": model,
                "Delta": r[METRIC] - base
            })

    delta = pd.DataFrame(rows)
    pivot = delta.pivot(index="Preprocessing", columns="Model", values="Delta").loc[ORDER]

    ax = pivot.plot(kind="bar")
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{dataset}: Δ{METRIC} vs baseline")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel(f"Δ{METRIC}")
    _add_bar_labels(ax, fmt="{:+.3f}")

    plt.tight_layout()
    out = os.path.join(DATA_DIR, out_name)
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def plot_line(df, dataset, out_name):
    """Line chart with point labels."""
    sub = df[df["Dataset"] == dataset]

    plt.figure()
    x = list(range(len(ORDER)))

    for model in sub["Model"].unique():
        y = sub[sub["Model"] == model].set_index("Preprocessing").loc[ORDER][METRIC].tolist()
        plt.plot(x, y, marker="o", label=model)

        # annotate points
        for xi, yi in zip(x, y):
            plt.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, ORDER)
    plt.title(f"{dataset}: {METRIC} Trend")
    plt.xlabel("Preprocessing")
    plt.ylabel(METRIC)
    plt.legend()

    plt.tight_layout()
    out = os.path.join(DATA_DIR, out_name)
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing: {CSV_PATH}. Run train.py first.")

    df = pd.read_csv(CSV_PATH)

    # Basic checks
    need_cols = {"Dataset", "Preprocessing", "Model", METRIC}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"results.csv missing columns: {missing}. Found: {df.columns.tolist()}")

    for dataset in ["IMDb", "Emotion"]:
        if dataset not in set(df["Dataset"].unique()):
            print(f"Warning: dataset '{dataset}' not found in results.csv. Skipping.")
            continue

        plot_bar(df, dataset, f"{dataset.lower()}_bar.png")
        plot_delta_bar(df, dataset, f"{dataset.lower()}_delta_bar.png")
        plot_line(df, dataset, f"{dataset.lower()}_line.png")

    print("\nDone. All figures are in:", DATA_DIR)


if __name__ == "__main__":
    main()
