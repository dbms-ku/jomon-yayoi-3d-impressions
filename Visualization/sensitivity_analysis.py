"""Sensitivity analysis of denoising hyperparameters (paper Figure 12).

Plots classification Accuracy and Macro-F1 on the Jomon-Yayoi dataset as
a function of each denoising hyperparameter (UE removal ratio alpha, ANR
break ratio beta, ANR survival ratio gamma), sweeping one parameter while
holding the others at their Bayesian-optimized values.

The arrays below are the actual sweep results reported in the paper: for
each parameter value the corresponding denoising pipeline (UE for alpha,
ANR for beta/gamma, WITHOUT the Noise Verification step) was applied to
the full Jomon-Yayoi dataset, followed by SA-PointNet++ classification at
8192 points; each entry is the mean over 30 inference trials (see
test_classification_sa.py). Reference anchors against Table 7:
alpha=0.01 -> 0.6746 (SA-PointNet++UE) and beta=0.20, gamma=0.10 ->
0.7249 (SA-PointNet++ANR).

Black-outlined markers indicate the Bayesian-optimized values used in all
other experiments (alpha*=0.01, beta*=0.20, gamma*=0.10).

To regenerate the underlying numbers, run the sweep:
    for each value v of the target parameter:
        denoise the Jomon-Yayoi data (ue_denoise.py / anr_denoise.py)
        evaluate with test_classification_sa.py (--repeat 30)
        record the mean Accuracy and Macro-F1
and update the arrays below (or pass --csv with columns
param,value,accuracy,macro_f1).
"""

import argparse
import csv

import matplotlib.pyplot as plt

# --- Sweep results (paper Figure 12) -----------------------------------------

# (a) UE removal ratio alpha, swept with SA-PointNet++ + UE.
ALPHA_VALUES = [0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03]
ALPHA_ACC = [0.6867, 0.6700, 0.6746, 0.6849, 0.6794, 0.6869, 0.6839,
             0.6866, 0.6889]
ALPHA_F1 = [0.5873, 0.5604, 0.5719, 0.5818, 0.5788, 0.5883, 0.5763,
            0.5758, 0.5955]
ALPHA_OPTIMAL = 0.01

# (b) ANR break ratio beta, swept with gamma fixed at 0.1
#     (SA-PointNet++ + ANR).
BETA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
BETA_ACC = [0.7149, 0.7297, 0.7386, 0.7249, 0.6589, 0.6430, 0.5894,
            0.5275, 0.5580]
BETA_F1 = [0.6220, 0.6426, 0.6449, 0.6102, 0.5224, 0.5087, 0.4404,
           0.3809, 0.4142]
BETA_OPTIMAL = 0.20

# (c) ANR survival ratio gamma, swept with beta fixed at 0.2
#     (SA-PointNet++ + ANR).
GAMMA_VALUES = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
GAMMA_ACC = [0.7154, 0.7106, 0.7025, 0.7249, 0.7051, 0.7212, 0.7080,
             0.7033, 0.7003]
GAMMA_F1 = [0.5933, 0.5874, 0.5899, 0.6102, 0.5930, 0.6098, 0.5871,
            0.5859, 0.5841]
GAMMA_OPTIMAL = 0.10

COLOR_ACC = "#E64B35"
COLOR_F1 = "#4DBBD5"


def load_sweeps_from_csv(csv_path):
    """Load sweep results from a CSV with columns param,value,accuracy,macro_f1.

    `param` must be one of alpha/beta/gamma. Returns a dict mapping the
    parameter name to (values, accuracies, macro_f1s).
    """
    sweeps = {"alpha": [], "beta": [], "gamma": []}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = row["param"].strip().lower()
            if key not in sweeps:
                raise ValueError(f"Unknown param '{row['param']}' in CSV")
            sweeps[key].append((float(row["value"]),
                                float(row["accuracy"]),
                                float(row["macro_f1"])))
    result = {}
    for key, rows in sweeps.items():
        rows.sort(key=lambda r: r[0])
        if rows:
            values, acc, f1 = zip(*rows)
            result[key] = (list(values), list(acc), list(f1))
    return result


def plot_sensitivity(ax, param_values, acc, f1, optimal,
                     param_name, param_symbol):
    """Plot Accuracy and Macro-F1 against a single parameter."""
    ax.plot(param_values, acc, "-o", color=COLOR_ACC, markersize=6,
            linewidth=1.8, label="Accuracy", zorder=3)
    ax.plot(param_values, f1, "-s", color=COLOR_F1, markersize=6,
            linewidth=1.8, label="Macro-F1", zorder=3)

    # Black-outlined markers at the Bayesian-optimized value.
    opt_idx = param_values.index(optimal)
    ax.axvline(x=optimal, color="gray", linestyle="--", alpha=0.5,
               linewidth=1)
    ax.plot(optimal, acc[opt_idx], "o", color=COLOR_ACC, markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=4)
    ax.plot(optimal, f1[opt_idx], "s", color=COLOR_F1, markersize=10,
            markeredgecolor="black", markeredgewidth=1.5, zorder=4)

    ax.annotate(f"{param_symbol}* = {optimal}",
                xy=(optimal, max(acc[opt_idx], f1[opt_idx])),
                xytext=(0, 18), textcoords="offset points",
                ha="center", fontsize=9, color="0.3",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel(f"{param_name} ({param_symbol})", fontsize=11, labelpad=8)
    ax.set_ylabel("Score", fontsize=11, labelpad=8)
    ax.set_ylim(0.45, 0.80)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the denoising-hyperparameter sensitivity "
                    "figure (paper Figure 12)."
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV (param,value,accuracy,macro_f1) "
                             "to plot instead of the embedded paper values.")
    parser.add_argument("--output", type=str, default="sensitivity_analysis",
                        help="Output path stem (writes .png and .pdf).")
    return parser.parse_args()


def main():
    args = parse_args()

    alpha = (ALPHA_VALUES, ALPHA_ACC, ALPHA_F1)
    beta = (BETA_VALUES, BETA_ACC, BETA_F1)
    gamma = (GAMMA_VALUES, GAMMA_ACC, GAMMA_F1)
    if args.csv:
        sweeps = load_sweeps_from_csv(args.csv)
        alpha = sweeps.get("alpha", alpha)
        beta = sweeps.get("beta", beta)
        gamma = sweeps.get("gamma", gamma)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    plot_sensitivity(axes[0], *alpha, ALPHA_OPTIMAL,
                     "Removal Ratio", r"$\alpha$")
    axes[0].set_title(r"(a) UE: Removal Ratio $\alpha$",
                      fontsize=11, fontweight="bold", pad=10)

    plot_sensitivity(axes[1], *beta, BETA_OPTIMAL,
                     "Break Ratio", r"$\beta$")
    axes[1].set_title(r"(b) ANR: Break Ratio $\beta$",
                      fontsize=11, fontweight="bold", pad=10)

    plot_sensitivity(axes[2], *gamma, GAMMA_OPTIMAL,
                     "Survival Ratio", r"$\gamma$")
    axes[2].set_title(r"(c) ANR: Survival Ratio $\gamma$",
                      fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout(w_pad=2.5)
    plt.savefig(f"{args.output}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{args.output}.pdf", bbox_inches="tight")
    print(f"Saved: {args.output}.png and {args.output}.pdf")


if __name__ == "__main__":
    main()