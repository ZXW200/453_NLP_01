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


def add_labels(ax, fmt="{:.3f}"):
    """Add numeric labels on bar tops."""
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, padding=3, fontsize=9)


def make_bar(df, ds, name):
    """Absolute bar chart with value labels."""
    sub = df[df["Dataset"] == ds]
    tbl = sub.pivot(index="Preprocessing", columns="Model", values=metric).loc[order]

    ax = tbl.plot(kind="bar")
    ax.set_title(f"{ds}: {metric} by Preprocessing")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel(metric)
    add_labels(ax, fmt="{:.3f}")

    plt.tight_layout()
    path = os.path.join(data_dir, name)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def make_delta_bar(df, ds, name):
    """Delta bar chart (relative to baseline) with +/- value labels."""
    sub = df[df["Dataset"] == ds].copy()

    lst = []
    for m in sub["Model"].unique():
        b = sub[(sub["Model"] == m) & (sub["Preprocessing"] == "baseline")][metric].iloc[0]
        for _, r in sub[sub["Model"] == m].iterrows():
            lst.append({
                "Preprocessing": r["Preprocessing"],
                "Model": m,
                "Delta": r[metric] - b
            })

    delta_df = pd.DataFrame(lst)
    tbl = delta_df.pivot(index="Preprocessing", columns="Model", values="Delta").loc[order]

    ax = tbl.plot(kind="bar")
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{ds}: Δ{metric} vs baseline")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel(f"Δ{metric}")
    add_labels(ax, fmt="{:+.3f}")

    plt.tight_layout()
    path = os.path.join(data_dir, name)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)


def make_line(df, ds, name):
    """Line chart with point labels."""
    sub = df[df["Dataset"] == ds]

    plt.figure()
    x = list(range(len(order)))

    for m in sub["Model"].unique():
        y = sub[sub["Model"] == m].set_index("Preprocessing").loc[order][metric].tolist()
        plt.plot(x, y, marker="o", label=m)

        # annotate points
        for i, j in zip(x, y):
            plt.text(i, j, f"{j:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, order)
    plt.title(f"{ds}: {metric} Trend")
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

    # Basic checks
    need = {"Dataset", "Preprocessing", "Model", metric}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"results.csv missing columns: {miss}. Found: {df.columns.tolist()}")

    for ds in ["IMDb", "Emotion"]:
        if ds not in set(df["Dataset"].unique()):
            print(f"Warning: dataset '{ds}' not found in results.csv. Skipping.")
            continue

        make_bar(df, ds, f"{ds.lower()}_bar.png")
        make_delta_bar(df, ds, f"{ds.lower()}_delta_bar.png")
        make_line(df, ds, f"{ds.lower()}_line.png")

    print("\nDone. All figures are in:", data_dir)


if __name__ == "__main__":
    main()
