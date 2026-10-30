"""Optuna-based hyperparameter optimization for SOR and ROR denoising.

This script optimizes the parameters of the two classical point-cloud
denoising baselines evaluated in the paper (Section 6.5):

  * Statistical Outlier Removal (SOR): nb_neighbors, std_ratio
  * Radius Outlier Removal (ROR): nb_points, radius (as a fraction of the
    object's maximum side length)

The objective is the mean Chamfer Distance between the denoised point
clouds and the expert-cleaned ground truth of the Noise-Pair Dataset.
Pairs are matched via the released dataset naming convention
(<stem>_noisy.stl in --noisy_dir and <stem>_clean.stl in --clean_dir).
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import optuna
from tqdm import tqdm

# --- Constants ---------------------------------------------------------------

# Number of points sampled from each mesh for the optimization objective.
DEFAULT_SAMPLE_POINTS = 30000

# Chamfer Distance (in mm) assigned when a filter removes every point,
# heavily penalizing configurations that destroy the geometry.
EMPTY_RESULT_PENALTY = 5.0

# Number of Optuna trials per method (paper Section 5.2 uses 100 trials).
DEFAULT_N_TRIALS = 100


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


# --- Data preparation ----------------------------------------------------------

def chamfer_distance(pcd_a, pcd_b):
    """Symmetric Chamfer Distance: mean(a->b) + mean(b->a), in mm."""
    d_ab = pcd_a.compute_point_cloud_distance(pcd_b)
    d_ba = pcd_b.compute_point_cloud_distance(pcd_a)
    return float(np.mean(d_ab) + np.mean(d_ba))


def prepare_optimization_data(noisy_dir, clean_dir,
                              noisy_suffix="_noisy", clean_suffix="_clean",
                              sample_points=DEFAULT_SAMPLE_POINTS):
    """Load noisy/clean pairs from the Noise-Pair Dataset.

    Files are matched by replacing noisy_suffix with clean_suffix in the
    file stem (e.g. NP_MW_0001_noisy.stl <-> NP_MW_0001_clean.stl).
    Returns a list of dicts with sampled point clouds and the noisy
    object's maximum bounding-box side length.
    """
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)

    noisy_files = sorted(noisy_dir.glob(f"*{noisy_suffix}.stl"))
    if not noisy_files:
        print(f"[Error] No '*{noisy_suffix}.stl' files found in {noisy_dir}.")
        return []

    dataset = []
    print("Loading noisy/clean pairs...")
    for noisy_path in tqdm(noisy_files):
        clean_name = noisy_path.name.replace(noisy_suffix, clean_suffix)
        clean_path = clean_dir / clean_name
        if not clean_path.exists():
            print(f"[Warning] Missing ground truth for {noisy_path.name}; "
                  "pair skipped.")
            continue

        mesh_noisy = o3d.io.read_triangle_mesh(str(noisy_path))
        mesh_clean = o3d.io.read_triangle_mesh(str(clean_path))
        if len(mesh_noisy.vertices) == 0 or len(mesh_clean.vertices) == 0:
            print(f"[Warning] Empty mesh in pair {noisy_path.name}; skipped.")
            continue

        pcd_noisy = mesh_noisy.sample_points_poisson_disk(
            number_of_points=sample_points
        )
        pcd_clean = mesh_clean.sample_points_poisson_disk(
            number_of_points=sample_points
        )

        bbox = pcd_noisy.get_axis_aligned_bounding_box()
        extent = bbox.get_max_bound() - bbox.get_min_bound()

        dataset.append({
            "pcd_noisy": pcd_noisy,
            "pcd_clean": pcd_clean,
            "object_max_size": float(np.max(extent)),
        })

    print(f"Loaded {len(dataset)} pairs.")
    return dataset


# --- Objectives ----------------------------------------------------------------

def objective_sor(trial, dataset):
    """Mean Chamfer Distance after Statistical Outlier Removal."""
    nb_neighbors = trial.suggest_int("nb_neighbors", 5, 50)
    std_ratio = trial.suggest_float("std_ratio", 0.5, 4.0)

    errors = []
    for data in dataset:
        filtered, _ = data["pcd_noisy"].remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        if len(filtered.points) == 0:
            errors.append(EMPTY_RESULT_PENALTY)
            continue
        errors.append(chamfer_distance(filtered, data["pcd_clean"]))

    return float(np.mean(errors))


def objective_ror(trial, dataset):
    """Mean Chamfer Distance after Radius Outlier Removal.

    The radius is parameterized relative to the object's maximum side
    length so that a single setting adapts to specimens of different size.
    """
    nb_points = trial.suggest_int("nb_points", 2, 30)
    radius_ratio = trial.suggest_float("radius_ratio", 0.01, 0.20)

    errors = []
    for data in dataset:
        radius = data["object_max_size"] * radius_ratio
        filtered, _ = data["pcd_noisy"].remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )
        if len(filtered.points) == 0:
            errors.append(EMPTY_RESULT_PENALTY)
            continue
        errors.append(chamfer_distance(filtered, data["pcd_clean"]))

    return float(np.mean(errors))


# --- CLI ------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize SOR/ROR denoising parameters via Optuna "
                    "(Chamfer Distance on the Noise-Pair Dataset)."
    )
    parser.add_argument(
        "--noisy_dir", type=str, default="./data/noise_pair/noisy",
        help="Directory containing the noisy Noise-Pair meshes.",
    )
    parser.add_argument(
        "--clean_dir", type=str, default="./data/noise_pair/clean",
        help="Directory containing the expert-cleaned ground-truth meshes.",
    )
    parser.add_argument(
        "--noisy_suffix", type=str, default="_noisy",
        help="Filename suffix identifying noisy files (default: _noisy).",
    )
    parser.add_argument(
        "--clean_suffix", type=str, default="_clean",
        help="Filename suffix identifying clean files (default: _clean).",
    )
    parser.add_argument(
        "--sample_points", type=int, default=DEFAULT_SAMPLE_POINTS,
        help="Points sampled per mesh for the objective computation.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=DEFAULT_N_TRIALS,
        help="Number of Optuna trials per method.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the Optuna sampler (for reproducible "
             "optimization runs).",
    )
    parser.add_argument(
        "--log_dir", type=str, default=".",
        help="Directory where the log file is written.",
    )
    return parser.parse_args()


def make_study(seed):
    sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else None
    return optuna.create_study(direction="minimize", sampler=sampler)


def main():
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_path = Path(args.log_dir) / f"sor_ror_optimization_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_fh:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.__stdout__, log_fh)
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            print(f"[INFO] Log file : {log_path}")
            print(f"[INFO] Timestamp: "
                  f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"[INFO] Noisy dir: {args.noisy_dir}")
            print(f"[INFO] Clean dir: {args.clean_dir}")
            if args.seed is not None:
                print(f"[INFO] Seed     : {args.seed}")

            dataset = prepare_optimization_data(
                args.noisy_dir, args.clean_dir,
                noisy_suffix=args.noisy_suffix,
                clean_suffix=args.clean_suffix,
                sample_points=args.sample_points,
            )
            if not dataset:
                print("[Error] No valid pairs loaded; aborting.")
                return

            # --- SOR -------------------------------------------------------
            print("\n[1/2] Optimizing SOR")
            study_sor = make_study(args.seed)
            study_sor.optimize(
                lambda trial: objective_sor(trial, dataset),
                n_trials=args.n_trials,
            )
            print(f"  Best nb_neighbors: "
                  f"{study_sor.best_params['nb_neighbors']}")
            print(f"  Best std_ratio   : "
                  f"{study_sor.best_params['std_ratio']:.4f}")
            print(f"  Minimum error    : {study_sor.best_value:.6f}")

            # --- ROR -------------------------------------------------------
            print("\n[2/2] Optimizing ROR")
            study_ror = make_study(args.seed)
            study_ror.optimize(
                lambda trial: objective_ror(trial, dataset),
                n_trials=args.n_trials,
            )
            best_ratio = study_ror.best_params["radius_ratio"]
            print(f"  Best nb_points   : {study_ror.best_params['nb_points']}")
            print(f"  Best radius_ratio: {best_ratio:.4f} "
                  f"({best_ratio * 100:.2f}% of the object's max length)")
            print(f"  Minimum error    : {study_ror.best_value:.6f}")

            # --- Summary ---------------------------------------------------
            print("\n" + "=" * 50)
            print("Optimization summary")
            print("=" * 50)
            print(f"SOR: nb_neighbors={study_sor.best_params['nb_neighbors']:2d}, "
                  f"std_ratio={study_sor.best_params['std_ratio']:.4f}, "
                  f"error={study_sor.best_value:.6f}")
            print(f"ROR: nb_points={study_ror.best_params['nb_points']:2d}, "
                  f"radius_ratio={study_ror.best_params['radius_ratio']:.4f}, "
                  f"error={study_ror.best_value:.6f}")
        finally:
            sys.stdout = original_stdout

    print(f"[INFO] Log saved to: {log_path}")


if __name__ == "__main__":
    main()