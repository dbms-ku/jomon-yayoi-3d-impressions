"""Optuna-based optimization of the Adaptive Neck Removal (ANR) parameters.

This script searches for the ANR break ratio beta and survival ratio gamma
(paper Section 5.2) that minimize the mean Chamfer Distance between
denoised point clouds and the expert-cleaned ground truth of the
Noise-Pair Dataset. The paper reports the optimized values beta* = 0.2 and
gamma* = 0.1.

Unlike the UE optimizer, each trial runs the full ANR pipeline including
the final geometry-recovery step, so the objective is computed on the
actual denoised high-resolution point cloud.

Run this script from the repository root so that anr_denoise.py is
importable (the denoising pipeline is imported from there to guarantee
that the optimized objective matches the deployed denoiser).
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
import optuna
from tqdm import tqdm

from anr_denoise import NUM_SAMPLE_POINTS, denoise_point_cloud

# --- Constants ---------------------------------------------------------------

# Chamfer Distance (in mm) assigned when denoising removes every point,
# heavily penalizing configurations that destroy the geometry.
EMPTY_RESULT_PENALTY = 5.0

# Number of Optuna trials (paper Section 5.2 reports 100 trials).
DEFAULT_N_TRIALS = 100

# Search ranges for the ANR parameters.
BREAK_RATIO_MIN, BREAK_RATIO_MAX = 0.10, 0.60
SURVIVAL_RATIO_MIN, SURVIVAL_RATIO_MAX = 0.05, 0.80


# --- Data preparation ----------------------------------------------------------

def chamfer_distance(pcd_a, pcd_b):
    """Symmetric Chamfer Distance: mean(a->b) + mean(b->a), in mm."""
    d_ab = pcd_a.compute_point_cloud_distance(pcd_b)
    d_ba = pcd_b.compute_point_cloud_distance(pcd_a)
    return float(np.mean(d_ab) + np.mean(d_ba))


def load_as_point_cloud(path, num_points=NUM_SAMPLE_POINTS):
    """Load a mesh or point cloud file and return an Open3D point cloud.

    Returns None if nothing usable could be loaded.
    """
    abs_path = str(Path(path).resolve())
    try:
        mesh = o3d.io.read_triangle_mesh(abs_path)
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            return mesh.sample_points_poisson_disk(num_points)
    except Exception:
        pass

    pcd = o3d.io.read_point_cloud(abs_path)
    if pcd is None or len(pcd.points) == 0:
        return None
    return pcd


def prepare_pairs(noisy_dir, clean_dir,
                  noisy_suffix="_noisy", clean_suffix="_clean"):
    """Load noisy/clean point-cloud pairs from the Noise-Pair Dataset.

    Files are matched by replacing noisy_suffix with clean_suffix in the
    file stem (e.g. NP_MW_0001_noisy.stl <-> NP_MW_0001_clean.stl).
    """
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)

    noisy_files = sorted(noisy_dir.glob(f"*{noisy_suffix}.stl"))
    if not noisy_files:
        print(f"[Error] No '*{noisy_suffix}.stl' files found in {noisy_dir}.")
        return []

    pairs = []
    print("Pre-loading pairs into memory...")
    for noisy_path in tqdm(noisy_files):
        clean_name = noisy_path.name.replace(noisy_suffix, clean_suffix)
        clean_path = clean_dir / clean_name
        if not clean_path.exists():
            print(f"[Warning] Missing ground truth for {noisy_path.name}; "
                  "pair skipped.")
            continue

        noisy_pcd = load_as_point_cloud(noisy_path)
        clean_pcd = load_as_point_cloud(clean_path)
        if noisy_pcd is None or clean_pcd is None:
            print(f"[Warning] Unreadable or empty pair {noisy_path.name}; "
                  "skipped.")
            continue

        pairs.append((noisy_pcd, clean_pcd))

    print(f"Loaded {len(pairs)} pairs.")
    return pairs


# --- Objective -------------------------------------------------------------------

def objective(trial, pairs):
    """Mean Chamfer Distance after ANR at the trial's parameters."""
    break_ratio = trial.suggest_float(
        "break_ratio", BREAK_RATIO_MIN, BREAK_RATIO_MAX
    )
    survival_ratio = trial.suggest_float(
        "survival_ratio", SURVIVAL_RATIO_MIN, SURVIVAL_RATIO_MAX
    )

    errors = []
    for noisy_pcd, clean_pcd in pairs:
        # denoise_point_cloud does not modify its input, so the preloaded
        # clouds can be reused across trials without copying.
        denoised = denoise_point_cloud(
            noisy_pcd, break_ratio=break_ratio, survival_ratio=survival_ratio
        )
        if denoised is None or len(denoised.points) == 0:
            errors.append(EMPTY_RESULT_PENALTY)
            continue

        errors.append(chamfer_distance(denoised, clean_pcd))

    return float(np.mean(errors))


# --- CLI ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize the ANR break/survival ratios via Optuna "
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
        "--n_trials", type=int, default=DEFAULT_N_TRIALS,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the Optuna sampler (for reproducible "
             "optimization runs).",
    )
    parser.add_argument(
        "--results_file", type=str, default="./optuna_anr_results.txt",
        help="Text file to which the best parameters are appended.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pairs = prepare_pairs(
        args.noisy_dir, args.clean_dir,
        noisy_suffix=args.noisy_suffix,
        clean_suffix=args.clean_suffix,
    )
    if not pairs:
        print("[Error] No valid pairs loaded; aborting.")
        return

    print("Starting Optuna optimization...")
    sampler = (optuna.samplers.TPESampler(seed=args.seed)
               if args.seed is not None else None)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, pairs),
                   n_trials=args.n_trials)

    print("\n" + "=" * 40)
    print("ANR optimization result:")
    print(f"  Best break_ratio (beta)    : "
          f"{study.best_params['break_ratio']:.4f}")
    print(f"  Best survival_ratio (gamma): "
          f"{study.best_params['survival_ratio']:.4f}")
    print(f"  Minimum error              : {study.best_value:.6f}")
    print("=" * 40)

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(f"Execution: {datetime.datetime.now()}\n")
        fh.write(f"Best params: {study.best_params}\n")
        fh.write(f"Best error : {study.best_value}\n\n")
    print(f"[INFO] Results appended to: {results_path}")


if __name__ == "__main__":
    main()