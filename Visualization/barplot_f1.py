"""Per-class F1 grouped bar chart (paper Figure 10).

Plots the per-class F1 scores on the Jomon-Yayoi dataset (8192 points)
for the three key configurations: the PointNet++ baseline, SA-PointNet++,
and SA-PointNet++ combined with ANR and Noise Verification.

The arrays below are the values reported in the paper (Figure 10). To
recompute them from your own evaluation, either derive per-class F1 from
per-run predictions,

    from sklearn.metrics import f1_score
    f1_per_class = f1_score(y_true, y_pred, average=None)

averaged over the 30 inference trials, or compute precision/recall from
the mean confusion matrix; the two aggregations differ slightly, so use
one consistently. Updated values can also be supplied via --csv with
columns class,pointnetpp,sa_pointnetpp,sa_anr_nv (one row per class, in
display order).
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

# --- Values reported in the paper (Figure 10) ---------------------------------

CLASSES = ["Foxtail\nMillet", "Rice", "Perilla", "Ailanthoides",
           "Maize\nWeevil"]

# PointNet++ baseline (no size, no denoising) - cf. Figure 9a.
F1_POINTNETPP = [0.1213, 0.6154, 0.1431, 0.2681, 0.6103]

# SA-PointNet++ (size-aware, no denoising) - cf. Figure 9b.
F1_SA_POINTNETPP = [0.8490, 0.6960, 0.2350, 0.5507, 0.7498]

# SA-PointNet++ + ANR + Noise Verification (best model) - cf. Figure 9c.
F1_SA_ANR_NV = [0.8394, 0.7259, 0.3107, 0.6231, 0.8016]

BAR_SPECS = [
    ("PointNet++", "#4878A8"),
    ("SA-PointNet++", "#E8833A"),
    ("SA-PointNet++ + ANR + NV", "#59A95A"),
]


def load_scores_from_csv(csv_path):
    """Load per-class F1 scores from a CSV.

    Expected columns: class, pointnetpp, sa_pointnetpp, sa_anr_nv.
    Rows are taken in file order as the display order.
    """
    classes, f1_pn, f1_sa, f1_nv = [], [], [], []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            classes.append(row["class"].replace("\\n", "\n"))
            f1_pn.append(float(row["pointnetpp"]))
            f1_sa.append(float(row["sa_pointnetpp"]))
            f1_nv.append(float(row["sa_anr_nv"]))
    return classes, f1_pn, f1_sa, f1_nv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the per-class F1 bar chart (paper Figure 10)."
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV (class,pointnetpp,sa_pointnetpp,"
                             "sa_anr_nv) to plot instead of the embedded "
                             "paper values.")
    parser.add_argument("--output", type=str, default="barplot_per_class_f1",
                        help="Output path stem (writes .png and .pdf).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.csv:
        classes, f1_pn, f1_sa, f1_nv = load_scores_from_csv(args.csv)
    else:
        classes = CLASSES
        f1_pn, f1_sa, f1_nv = F1_POINTNETPP, F1_SA_POINTNETPP, F1_SA_ANR_NV

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    all_bars = []
    for offset, scores, (label, color) in zip(
        (-width, 0.0, width), (f1_pn, f1_sa, f1_nv), BAR_SPECS
    ):
        bars = ax.bar(x + offset, scores, width, label=label,
                      color=color, edgecolor="white", linewidth=0.5)
        all_bars.append(bars)

    ax.set_xlabel("Class", fontsize=12, labelpad=10)
    ax.set_ylabel("F1 Score", fontsize=12, labelpad=10)
    ax.set_title("Per-Class F1 Scores on J\u014dmon-Yayoi Dataset "
                 "(8192 points)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels above each bar.
    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color="0.3")

    plt.tight_layout()
    plt.savefig(f"{args.output}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{args.output}.pdf", bbox_inches="tight")
    print(f"Saved: {args.output}.png and {args.output}.pdf")


if __name__ == "__main__":
    main()