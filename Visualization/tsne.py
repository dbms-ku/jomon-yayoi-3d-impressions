"""t-SNE feature-embedding visualization (paper Figure 11).

Renders side-by-side t-SNE plots comparing the PointNet++ and SA-PointNet++
global feature spaces, overlaying the Experimental test set (circles) and
the Jomon-Yayoi dataset (triangles), colored by class.

The script consumes precomputed feature files (.npz). Each file must
contain a feature matrix and a label vector; the array keys are
configurable (default: 'features' and 'labels'). Extract features from the
global descriptor after max-pooling: 1024-dim for PointNet++, or the
1027-dim concatenation with the AABB size feature for SA-PointNet++ (see
the size-aware model's feature-export utility).

Four files are required, one per (model, dataset) pair:
  * PointNet++      on Experimental / on Jomon-Yayoi
  * SA-PointNet++   on Experimental / on Jomon-Yayoi
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- Constants ---------------------------------------------------------------

CLASS_NAMES = ["Foxtail Millet", "Rice", "Perilla", "Ailanthoides",
               "Maize Weevil"]
N_CLASSES = len(CLASS_NAMES)

# One color per class, shared across both panels.
CLASS_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#8491B4"]

# Dataset flags used internally.
DATASET_EXP = 0
DATASET_JY = 1


# --- Data loading ------------------------------------------------------------

def load_features(npz_path, feature_key, label_key):
    """Load a feature matrix and label vector from an .npz file."""
    data = np.load(npz_path)
    if feature_key not in data or label_key not in data:
        raise KeyError(
            f"{npz_path} must contain '{feature_key}' and '{label_key}'; "
            f"found keys: {list(data.keys())}"
        )
    return data[feature_key], data[label_key]


def combine_datasets(exp_path, jy_path, feature_key, label_key):
    """Load Experimental and Jomon-Yayoi features and tag their origin."""
    exp_feats, exp_labels = load_features(exp_path, feature_key, label_key)
    jy_feats, jy_labels = load_features(jy_path, feature_key, label_key)

    features = np.concatenate([exp_feats, jy_feats])
    labels = np.concatenate([exp_labels, jy_labels])
    datasets = np.concatenate([
        np.full(len(exp_feats), DATASET_EXP),
        np.full(len(jy_feats), DATASET_JY),
    ])
    return features, labels, datasets, len(exp_feats), len(jy_feats)


# --- t-SNE -------------------------------------------------------------------

def compute_tsne(features, perplexity, random_state, n_iter):
    """Standardize features and project them to 2D with t-SNE."""
    scaled = StandardScaler().fit_transform(features)
    # scikit-learn renamed n_iter -> max_iter; support both.
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=random_state, max_iter=n_iter)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=random_state, n_iter=n_iter)
    return tsne.fit_transform(scaled)


# --- Plotting ----------------------------------------------------------------

def plot_panel(ax, embedding, labels, datasets, title):
    """Draw one t-SNE panel: Experimental as circles, Jomon-Yayoi as triangles."""
    exp_mask = datasets == DATASET_EXP
    jy_mask = datasets == DATASET_JY

    for c in range(N_CLASSES):
        mask_exp = exp_mask & (labels == c)
        if np.any(mask_exp):
            ax.scatter(embedding[mask_exp, 0], embedding[mask_exp, 1],
                       c=CLASS_COLORS[c], marker="o", s=60, alpha=0.85,
                       edgecolors="white", linewidths=0.5, zorder=3)
        mask_jy = jy_mask & (labels == c)
        if np.any(mask_jy):
            ax.scatter(embedding[mask_jy, 0], embedding[mask_jy, 1],
                       c=CLASS_COLORS[c], marker="^", s=45, alpha=0.55,
                       edgecolors="gray", linewidths=0.3, zorder=2)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, linestyle="--")


def add_legend(fig):
    """Add a unified legend: class colors plus dataset markers."""
    class_handles = [
        mlines.Line2D([], [], color=CLASS_COLORS[c], marker="s",
                      linestyle="None", markersize=8, label=CLASS_NAMES[c])
        for c in range(N_CLASSES)
    ]
    dataset_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=8, markerfacecolor="gray", alpha=0.85,
                      label="Experimental"),
        mlines.Line2D([], [], color="gray", marker="^", linestyle="None",
                      markersize=8, markerfacecolor="gray", alpha=0.55,
                      label="Jomon-Yayoi"),
    ]
    spacer = mlines.Line2D([], [], linestyle="None")
    handles = class_handles + [spacer] + dataset_handles
    labels = [h.get_label() for h in handles]
    labels[N_CLASSES] = ""  # blank spacer label

    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=9.5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.5, handletextpad=0.5)


# --- CLI ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render side-by-side t-SNE plots of PointNet++ and "
                    "SA-PointNet++ feature spaces (paper Figure 11)."
    )
    parser.add_argument("--pn_exp", required=True,
                        help="PointNet++ features on the Experimental set (.npz).")
    parser.add_argument("--pn_jy", required=True,
                        help="PointNet++ features on the Jomon-Yayoi set (.npz).")
    parser.add_argument("--sa_exp", required=True,
                        help="SA-PointNet++ features on the Experimental set (.npz).")
    parser.add_argument("--sa_jy", required=True,
                        help="SA-PointNet++ features on the Jomon-Yayoi set (.npz).")
    parser.add_argument("--feature_key", default="features",
                        help="Array key for the feature matrix (default: features).")
    parser.add_argument("--label_key", default="labels",
                        help="Array key for the label vector (default: labels).")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity (paper: 30).")
    parser.add_argument("--random_state", type=int, default=42,
                        help="t-SNE random seed.")
    parser.add_argument("--n_iter", type=int, default=1000,
                        help="t-SNE iterations.")
    parser.add_argument("--output", default="tsne_comparison",
                        help="Output path stem (writes .png and .pdf).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading PointNet++ features...")
    feats_pn, labels_pn, dsets_pn, n_exp, n_jy = combine_datasets(
        args.pn_exp, args.pn_jy, args.feature_key, args.label_key)
    print(f"  Experimental: {n_exp}, Jomon-Yayoi: {n_jy}")

    print("Loading SA-PointNet++ features...")
    feats_sa, labels_sa, dsets_sa, _, _ = combine_datasets(
        args.sa_exp, args.sa_jy, args.feature_key, args.label_key)

    print("Computing t-SNE for PointNet++ features...")
    embed_pn = compute_tsne(feats_pn, args.perplexity,
                            args.random_state, args.n_iter)
    print("Computing t-SNE for SA-PointNet++ features...")
    embed_sa = compute_tsne(feats_sa, args.perplexity,
                            args.random_state, args.n_iter)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plot_panel(ax1, embed_pn, labels_pn, dsets_pn,
               "(a) PointNet++ (shape only)")
    plot_panel(ax2, embed_sa, labels_sa, dsets_sa,
               "(b) SA-PointNet++ (shape + size)")
    add_legend(fig)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(f"{args.output}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{args.output}.pdf", bbox_inches="tight")
    print(f"Saved: {args.output}.png and {args.output}.pdf")


if __name__ == "__main__":
    main()