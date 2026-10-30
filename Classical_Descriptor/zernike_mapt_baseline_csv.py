"""Classical descriptor baselines: 3D Zernike moments and MAPT.

Classical baselines for the paper (Section 6.4, Table 6). This script
extends the Zernike-only baseline with a Multiscale Area Projection
Transform (MAPT) descriptor and its combinations. Descriptors, optionally
augmented with axis-aligned bounding-box (AABB) size features, are
classified with SVM (RBF/linear) and Random Forest. The paper reports
Random Forest results; all three classifiers are computed here for
completeness.

Note on MAPT: the implementation below is a multiscale spherical-shell
projection histogram used as the "MAPT" descriptor in this project. If a
strict reproduction of the original Multiscale Area Projection Transform
is required, replace compute_mapt_descriptor with a reference
implementation; the difference is documented in the repository README.

Data specification:
  * Experimental data: split (train/valid/test) is given by a CSV with
    columns 'class', 'split', 'path'.
  * Jomon-Yayoi data: all files under --jy_root (one subfolder per class)
    are used as a held-out test set.

Class subfolders and the CSV 'class' column may use either the English
class codes (foxtail_millet, rice, perilla, ailanthoides, maize_weevil)
or the corresponding original-language folder names (see CLASS_ALIASES).

Requirements:
    pip install numpy scipy trimesh scikit-learn pandas
    pip install ZMPY3D   # optional; falls back to a NumPy implementation

Note on reproducibility: surface sampling is seeded via --seed so that the
extracted descriptors are reproducible across runs.
"""

import argparse
import datetime
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# --- Logging utility ---------------------------------------------------------

class Tee:
    """Duplicate writes to multiple streams (console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def fileno(self):
        return self._streams[0].fileno()


# --- Class naming ------------------------------------------------------------

CLASS_NAMES = [
    "foxtail_millet", "rice", "perilla", "ailanthoides", "maize_weevil",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Accept alternative (original-language) folder / label names as aliases for
# the canonical English class codes.
CLASS_ALIASES = {
    "アワ": "foxtail_millet",
    "イネ": "rice",
    "エゴマ": "perilla",
    "カラスザンショウ": "ailanthoides",
    "コクゾウムシ": "maize_weevil",
}


def resolve_class(name):
    """Map a class label (English code or alias) to its canonical code."""
    name = str(name).strip()
    if name in CLASS_TO_IDX:
        return name
    return CLASS_ALIASES.get(name)


# --- 3D Zernike moments ------------------------------------------------------

try:
    import ZMPY3D as zm
    HAS_ZMPY3D = True
    print("[INFO] Using ZMPY3D for Zernike computation.")
except ImportError:
    HAS_ZMPY3D = False
    print("[INFO] ZMPY3D not found; using the built-in NumPy implementation.")


def _factorial(n):
    result = np.ones(n + 1, dtype=np.float64)
    for i in range(2, n + 1):
        result[i] = result[i - 1] * i
    return result


def _radial_polynomial_3d(n, l, r):
    if (n - l) % 2 != 0:
        return np.zeros_like(r)
    k = (n - l) // 2
    facts = _factorial(n + l + 10)
    result = np.zeros_like(r)
    for s in range(k + 1):
        num = ((-1) ** s) * facts[n - s]
        den = facts[s] * facts[k - s] * facts[(n + l) // 2 + 1 + s]
        if den == 0:
            continue
        result = result + (num / den) * (r ** (n - 2 * s))
    return result


def num_zernike_descriptors(max_order):
    return sum(1 for n in range(max_order + 1)
               for l in range(n + 1) if (n - l) % 2 == 0)


def compute_zernike_moments_voxel(voxel_grid, max_order=20):
    """NumPy fallback: rotation-invariant 3D Zernike descriptors."""
    try:
        from scipy.special import sph_harm_y

        def _sph_harm(m, l, phi, theta):
            return sph_harm_y(l, m, theta, phi)
    except ImportError:
        from scipy.special import sph_harm

        def _sph_harm(m, l, phi, theta):
            return sph_harm(m, l, phi, theta)

    grid_size = voxel_grid.shape[0]
    coords = np.linspace(-1, 1, grid_size)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    theta = np.arccos(np.clip(zz / (r + 1e-12), -1, 1))
    phi = np.arctan2(yy, xx)
    mask = (r <= 1.0) & (voxel_grid > 0)

    if mask.sum() == 0:
        return np.zeros(num_zernike_descriptors(max_order))

    r_m, theta_m, phi_m = r[mask], theta[mask], phi[mask]
    descriptors = []
    for n in range(max_order + 1):
        for l in range(n + 1):
            if (n - l) % 2 != 0:
                continue
            R_nl = _radial_polynomial_3d(n, l, r_m)
            moment_norm_sq = 0.0
            for m in range(-l, l + 1):
                Y_lm = _sph_harm(m, l, phi_m, theta_m)
                omega = np.sum(R_nl * np.conj(Y_lm)) * (2.0 / grid_size) ** 3
                moment_norm_sq += np.abs(omega) ** 2
            descriptors.append(np.sqrt(moment_norm_sq))
    return np.array(descriptors)


def compute_zernike(voxel_grid, max_order=20):
    if not HAS_ZMPY3D:
        return compute_zernike_moments_voxel(voxel_grid, max_order)
    try:
        geo_m = zm.get_geometric_moments_from_voxel(
            voxel_grid.astype(np.float64), max_order
        )
        z_m = zm.convert_geometric_to_zernike(geo_m, max_order)
        return zm.get_3dzd(z_m, max_order)
    except Exception as exc:
        print(f"[WARN] ZMPY3D failed ({exc}); using NumPy fallback.")
        return compute_zernike_moments_voxel(voxel_grid, max_order)


# --- MAPT descriptor ---------------------------------------------------------

def compute_mapt_descriptor(points, n_scales=4, n_bins_elevation=8,
                            n_bins_azimuth=16, normalize=True):
    """Multiscale spherical-shell projection histogram (MAPT).

    The object is centered and normalized into the unit sphere. For each of
    n_scales radial shells, the enclosed surface points are projected onto
    an (elevation x azimuth) histogram; concatenating the shells captures
    multiscale shape information.
    """
    mapt_dim = n_scales * n_bins_elevation * n_bins_azimuth
    if len(points) == 0:
        return np.zeros(mapt_dim)

    pts = points - points.mean(axis=0)
    max_dist = np.linalg.norm(pts, axis=1).max()
    if max_dist > 1e-8:
        pts = pts / max_dist

    r = np.linalg.norm(pts, axis=1)
    theta = np.arccos(np.clip(pts[:, 2] / (r + 1e-12), -1, 1))
    phi = np.arctan2(pts[:, 1], pts[:, 0])

    boundaries = np.linspace(0.0, 1.0, n_scales + 1)
    parts = []
    for i in range(n_scales):
        r_inner, r_outer = boundaries[i], boundaries[i + 1]
        if i == n_scales - 1:
            mask = (r >= r_inner) & (r <= r_outer)
        else:
            mask = (r >= r_inner) & (r < r_outer)

        if mask.sum() == 0:
            parts.append(np.zeros(n_bins_elevation * n_bins_azimuth))
            continue

        hist, _, _ = np.histogram2d(
            theta[mask], phi[mask],
            bins=[n_bins_elevation, n_bins_azimuth],
            range=[[0, np.pi], [-np.pi, np.pi]],
        )
        hist = hist.flatten().astype(np.float64)
        if normalize and hist.sum() > 0:
            hist /= hist.sum()
        parts.append(hist)

    return np.concatenate(parts)


# --- Mesh -> points / voxels -------------------------------------------------

def load_stl_to_points(stl_path, n_points=50000, seed=None):
    """Sample points from an STL surface (optionally seeded)."""
    import trimesh
    mesh = trimesh.load(stl_path, force="mesh")
    if seed is not None:
        points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    else:
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def voxelize_points(points, grid_size=64):
    mins, maxs = points.min(axis=0), points.max(axis=0)
    aabb_size = maxs - mins
    extent = maxs - mins
    extent[extent == 0] = 1.0
    normalized = (points - mins) / extent * (grid_size - 1)
    indices = np.clip(normalized.astype(int), 0, grid_size - 1)
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    voxel_grid = binary_fill_holes(voxel_grid).astype(np.uint8)
    return voxel_grid, aabb_size


# --- Data loading ------------------------------------------------------------

SPLIT_ALIASES = {"train": "train", "valid": "valid",
                 "val": "valid", "test": "test"}


def load_split_from_csv(csv_path, base_dir=None):
    """Load train/valid/test paths and labels from a split CSV."""
    df = pd.read_csv(csv_path)
    required = {"class", "split", "path"}
    missing = required - set(df.columns.str.strip().str.lower())
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    df.columns = df.columns.str.strip().str.lower()
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(csv_path))

    splits = {sp: {"paths": [], "labels": []}
              for sp in ("train", "valid", "test")}
    skipped_class, skipped_split, skipped_file = [], [], []

    for _, row in df.iterrows():
        canonical = resolve_class(row["class"])
        if canonical is None:
            skipped_class.append(str(row["class"]).strip())
            continue
        label = CLASS_TO_IDX[canonical]

        split_key = SPLIT_ALIASES.get(str(row["split"]).strip().lower())
        if split_key is None:
            skipped_split.append(str(row["split"]).strip())
            continue

        raw_path = str(row["path"]).strip()
        abs_path = raw_path if os.path.isabs(raw_path) else \
            os.path.normpath(os.path.join(base_dir, raw_path))
        if not os.path.exists(abs_path):
            skipped_file.append(abs_path)
            continue

        splits[split_key]["paths"].append(abs_path)
        splits[split_key]["labels"].append(label)

    if skipped_class:
        print(f"  [WARN] Unknown class labels (skipped): "
              f"{sorted(set(skipped_class))}")
    if skipped_split:
        print(f"  [WARN] Unknown split values (skipped): "
              f"{sorted(set(skipped_split))}")
    if skipped_file:
        print(f"  [WARN] Skipped {len(skipped_file)} rows with missing files.")

    print(f"  CSV: {os.path.basename(csv_path)}")
    for sp in ("train", "valid", "test"):
        n = len(splits[sp]["paths"])
        if n:
            print(f"    {sp:5s}: {n} samples")
    return splits


def load_dataset_from_dir(data_dir):
    """Load all STL files under per-class subfolders as a test set."""
    paths, labels = [], []
    for class_name in CLASS_NAMES:
        candidate_dirs = [class_name] + [
            alias for alias, canon in CLASS_ALIASES.items()
            if canon == class_name
        ]
        class_dir = None
        for name in candidate_dirs:
            if os.path.isdir(os.path.join(data_dir, name)):
                class_dir = os.path.join(data_dir, name)
                break
        if class_dir is None:
            print(f"  [WARN] Missing class folder (skipped): {class_name}")
            continue

        stl_files = sorted(
            glob.glob(os.path.join(class_dir, "*.stl"))
            + glob.glob(os.path.join(class_dir, "*.STL"))
        )
        for f in stl_files:
            paths.append(f)
            labels.append(CLASS_TO_IDX[class_name])

    print(f"  Directory: {data_dir}")
    print(f"  Total: {len(paths)} samples (all used for testing)")
    return paths, labels


# --- Feature extraction ------------------------------------------------------

def extract_all_features(file_paths, max_order=20, grid_size=64,
                         n_points=50000, mapt_scales=4, mapt_elev=8,
                         mapt_azim=16, seed=None):
    """Extract Zernike, AABB-size and MAPT features for a list of files."""
    n_zernike = num_zernike_descriptors(max_order)
    mapt_dim = mapt_scales * mapt_elev * mapt_azim
    zernike, aabb, mapt = [], [], []

    for i, fpath in enumerate(file_paths):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(file_paths)}: "
                  f"{os.path.basename(fpath)}")
        try:
            file_seed = None if seed is None else seed + i
            points = load_stl_to_points(fpath, n_points=n_points,
                                        seed=file_seed)
            voxel_grid, aabb_size = voxelize_points(points, grid_size=grid_size)
            zernike.append(compute_zernike(voxel_grid, max_order))
            aabb.append(aabb_size)
            mapt.append(compute_mapt_descriptor(
                points, n_scales=mapt_scales,
                n_bins_elevation=mapt_elev, n_bins_azimuth=mapt_azim,
            ))
        except Exception as exc:
            print(f"  [ERROR] Failed on {fpath}: {exc}")
            zernike.append(np.zeros(n_zernike))
            aabb.append(np.zeros(3))
            mapt.append(np.zeros(mapt_dim))

    return np.array(zernike), np.array(aabb), np.array(mapt)


# --- Classification ----------------------------------------------------------

def run_classification(X_train, y_train, X_test, y_test,
                       classifier_name="rf", seed=42):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if classifier_name == "svm":
        clf = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)
    elif classifier_name == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                     n_jobs=-1)
    elif classifier_name == "svm_linear":
        clf = SVC(kernel="linear", C=1.0, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred,
                                         target_names=CLASS_NAMES, digits=4),
    }


# --- CLI ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classical descriptor baselines (Zernike + MAPT)."
    )
    parser.add_argument("--exp_csv", type=str, required=True,
                        help="Experimental split CSV (class/split/path).")
    parser.add_argument("--jy_root", type=str, required=True,
                        help="Jomon-Yayoi root (per-class subfolders).")
    parser.add_argument("--exp_base_dir", type=str, default=None,
                        help="Base dir for relative CSV paths "
                             "(default: the CSV's directory).")
    parser.add_argument("--max_order", type=int, default=20,
                        help="Zernike maximum order.")
    parser.add_argument("--grid_size", type=int, default=64,
                        help="Voxel grid size for Zernike.")
    parser.add_argument("--n_points", type=int, default=50000,
                        help="Points sampled per mesh.")
    parser.add_argument("--mapt_scales", type=int, default=4,
                        help="MAPT: number of radial shells.")
    parser.add_argument("--mapt_elev", type=int, default=8,
                        help="MAPT: elevation bins.")
    parser.add_argument("--mapt_azim", type=int, default=16,
                        help="MAPT: azimuth bins.")
    parser.add_argument("--include_size", action="store_true",
                        help="Also evaluate descriptor + AABB size features.")
    parser.add_argument("--include_mapt", action="store_true",
                        help="Also evaluate MAPT and Zernike+MAPT features.")
    parser.add_argument("--classifiers", type=str, nargs="+",
                        default=["rf", "svm", "svm_linear"],
                        choices=["rf", "svm", "svm_linear"],
                        help="Classifiers to evaluate (paper reports rf).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling and classifiers.")
    parser.add_argument("--save_features", type=str, default=None,
                        help="Path to cache extracted features (.npz).")
    parser.add_argument("--load_features", type=str, default=None,
                        help="Path to load cached features (.npz).")
    parser.add_argument("--log_dir", type=str, default=".",
                        help="Directory where the log file is written.")
    return parser.parse_args()


def build_feature_configs(feats, include_size, include_mapt):
    """Assemble the feature combinations to evaluate."""
    (tr_z, tr_a, tr_m), (te_z, te_a, te_m), (jy_z, jy_a, jy_m) = feats
    configs = {
        "Zernike only": {"X_train": tr_z, "X_test_exp": te_z, "X_test_jy": jy_z},
    }
    if include_size:
        configs["Zernike + AABB size"] = {
            "X_train": np.hstack([tr_z, tr_a]),
            "X_test_exp": np.hstack([te_z, te_a]),
            "X_test_jy": np.hstack([jy_z, jy_a]),
        }
    if include_mapt:
        configs["MAPT only"] = {
            "X_train": tr_m, "X_test_exp": te_m, "X_test_jy": jy_m}
        configs["MAPT + AABB size"] = {
            "X_train": np.hstack([tr_m, tr_a]),
            "X_test_exp": np.hstack([te_m, te_a]),
            "X_test_jy": np.hstack([jy_m, jy_a])}
        configs["Zernike + MAPT"] = {
            "X_train": np.hstack([tr_z, tr_m]),
            "X_test_exp": np.hstack([te_z, te_m]),
            "X_test_jy": np.hstack([jy_z, jy_m])}
        if include_size:
            configs["Zernike + MAPT + AABB size"] = {
                "X_train": np.hstack([tr_z, tr_m, tr_a]),
                "X_test_exp": np.hstack([te_z, te_m, te_a]),
                "X_test_jy": np.hstack([jy_z, jy_m, jy_a])}
    return configs


def main():
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(args.log_dir,
                            f"zernike_mapt_baseline_{timestamp}.log")
    os.makedirs(args.log_dir, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.__stdout__, log_fh)

    try:
        print(f"[INFO] Log file: {log_path}")
        print(f"[INFO] Timestamp: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 70)
        print("Classical descriptor baselines (Zernike + MAPT)")
        print(f"  max_order={args.max_order}, grid_size={args.grid_size}, "
              f"n_points={args.n_points}, seed={args.seed}")
        print(f"  MAPT: scales={args.mapt_scales}, elev={args.mapt_elev}, "
              f"azim={args.mapt_azim}")
        print(f"  include_size={args.include_size}, "
              f"include_mapt={args.include_mapt}")
        print("=" * 70)

        mapt_dim = args.mapt_scales * args.mapt_elev * args.mapt_azim

        if args.load_features and os.path.exists(args.load_features):
            print(f"\n[LOAD] {args.load_features}")
            d = np.load(args.load_features, allow_pickle=True)
            if "exp_train_mapt" not in d.files:
                raise KeyError(
                    "Cached features lack MAPT arrays; re-extract with "
                    "--save_features to use MAPT.")
            exp_train = (d["exp_train_feat"], d["exp_train_aabb"],
                         d["exp_train_mapt"])
            exp_test = (d["exp_test_feat"], d["exp_test_aabb"],
                        d["exp_test_mapt"])
            jy_test = (d["jy_test_feat"], d["jy_test_aabb"], d["jy_test_mapt"])
            exp_train_label = d["exp_train_label"]
            exp_test_label = d["exp_test_label"]
            jy_test_label = d["jy_test_label"]
        else:
            print(f"\n[1] Loading Experimental CSV: {args.exp_csv}")
            exp_splits = load_split_from_csv(args.exp_csv,
                                             base_dir=args.exp_base_dir)
            print(f"\n[2] Loading Jomon-Yayoi directory: {args.jy_root}")
            jy_paths, jy_labels = load_dataset_from_dir(args.jy_root)
            if not jy_paths:
                raise RuntimeError(f"No STL files found under {args.jy_root}")

            kw = dict(max_order=args.max_order, grid_size=args.grid_size,
                      n_points=args.n_points, mapt_scales=args.mapt_scales,
                      mapt_elev=args.mapt_elev, mapt_azim=args.mapt_azim,
                      seed=args.seed)

            def extract_split(split_key):
                paths = exp_splits[split_key]["paths"]
                labels = exp_splits[split_key]["labels"]
                if not paths:
                    n_z = num_zernike_descriptors(args.max_order)
                    return ((np.zeros((0, n_z)), np.zeros((0, 3)),
                             np.zeros((0, mapt_dim))), np.array([], dtype=int))
                print(f"\n[extract] Experimental {split_key}: "
                      f"{len(paths)} samples")
                return (extract_all_features(paths, **kw), np.array(labels))

            exp_train, exp_train_label = extract_split("train")
            _, _ = extract_split("valid")  # extracted for completeness
            exp_test, exp_test_label = extract_split("test")

            print(f"\n[extract] Jomon-Yayoi: {len(jy_paths)} samples")
            jy_test = extract_all_features(jy_paths, **kw)
            jy_test_label = np.array(jy_labels)

            if args.save_features:
                np.savez(
                    args.save_features,
                    exp_train_feat=exp_train[0], exp_train_aabb=exp_train[1],
                    exp_train_mapt=exp_train[2], exp_train_label=exp_train_label,
                    exp_test_feat=exp_test[0], exp_test_aabb=exp_test[1],
                    exp_test_mapt=exp_test[2], exp_test_label=exp_test_label,
                    jy_test_feat=jy_test[0], jy_test_aabb=jy_test[1],
                    jy_test_mapt=jy_test[2], jy_test_label=jy_test_label,
                )
                print(f"\n[SAVE] Features cached to {args.save_features}")

        configs = build_feature_configs(
            (exp_train, exp_test, jy_test),
            args.include_size, args.include_mapt,
        )

        clf_display = {"svm": "SVM (RBF)", "rf": "Random Forest",
                       "svm_linear": "SVM (Linear)"}
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        summary = []
        for feat_name, data in configs.items():
            print(f"\n--- Feature: {feat_name} "
                  f"(dim={data['X_train'].shape[1]}) ---")
            for clf_name in args.classifiers:
                print(f"\n  Classifier: {clf_display[clf_name]}")
                if len(exp_test_label) > 0:
                    res_exp = run_classification(
                        data["X_train"], exp_train_label,
                        data["X_test_exp"], exp_test_label,
                        classifier_name=clf_name, seed=args.seed)
                    print(f"    [Experimental] Acc: {res_exp['accuracy']:.4f} "
                          f"Macro-F1: {res_exp['macro_f1']:.4f}")
                else:
                    res_exp = {"accuracy": float("nan"),
                               "macro_f1": float("nan")}
                    print("    [Experimental] (no test split)")

                res_jy = run_classification(
                    data["X_train"], exp_train_label,
                    data["X_test_jy"], jy_test_label,
                    classifier_name=clf_name, seed=args.seed)
                print(f"    [Jomon-Yayoi]  Acc: {res_jy['accuracy']:.4f} "
                      f"Macro-F1: {res_jy['macro_f1']:.4f}")

                summary.append({
                    "feature": feat_name,
                    "classifier": clf_display[clf_name],
                    "exp_acc": res_exp["accuracy"],
                    "exp_f1": res_exp["macro_f1"],
                    "jy_acc": res_jy["accuracy"],
                    "jy_f1": res_jy["macro_f1"],
                })

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Method':<40} {'ExpAcc':>8} {'ExpF1':>8} "
              f"{'JYAcc':>8} {'JYF1':>8}")
        print("-" * 70)
        for r in summary:
            method = f"{r['feature']} + {r['classifier']}"
            ea = f"{r['exp_acc']:>8.4f}" if not np.isnan(r['exp_acc']) \
                else "     ---"
            ef = f"{r['exp_f1']:>8.4f}" if not np.isnan(r['exp_f1']) \
                else "     ---"
            print(f"{method:<40} {ea} {ef} "
                  f"{r['jy_acc']:>8.4f} {r['jy_f1']:>8.4f}")
    finally:
        sys.stdout = original_stdout
        log_fh.close()

    print(f"[INFO] Log saved to: {log_path}")


if __name__ == "__main__":
    main()