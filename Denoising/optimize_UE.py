"""Optuna-based optimization of the Uniform Erosion (UE) removal ratio.

This script searches for the UE removal ratio alpha (paper Section 5.2)
that minimizes the mean Chamfer Distance between denoised point clouds and
the expert-cleaned ground truth of the Noise-Pair Dataset. The paper
reports the optimized value alpha* = 0.01.

To keep each trial fast, the voxelization and solidification of every
specimen are precomputed once, and only the erosion depth varies across
trials. The Chamfer Distance is computed between the restored voxel
centers and the ground truth, i.e. the final geometry-recovery step of the
deployed pipeline (selecting original high-resolution points) is skipped;
this approximation is consistent across trials and therefore does not
affect the ranking of candidate parameters.

Run this script from the repository root so that ue_denoise.py is
importable (the erosion-depth formula is imported from there to guarantee
that the optimized objective matches the deployed denoiser).
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import optuna
import scipy.ndimage as ndimage
from tqdm import tqdm

from ue_denoise import (
    SOLIDIFY_ITERATIONS,
    STRUCT_3D,
    VOXEL_RES_FACTOR,
    compute_erosion_iterations,
    voxelize,
)

# --- Constants ---------------------------------------------------------------

# Number of points sampled from each mesh for the optimization objective.
DEFAULT_SAMPLE_POINTS = 30000

# Chamfer Distance (in mm) assigned when erosion removes every voxel,
# heavily penalizing configurations that destroy the geometry.
EMPTY_RESULT_PENALTY = 5.0

# Number of Optuna trials (paper Section 5.2 reports 100 trials).
DEFAULT_N_TRIALS = 100

# Search range for the removal ratio alpha: 0.1% to 5% of the object's
# maximum side length.
ALPHA_SEARCH_MIN = 0.001
ALPHA_SEARCH_MAX = 0.050


# --- Data preparation ----------------------------------------------------------

def chamfer_distance(pcd_a, pcd_b):
    """Symmetric Chamfer Distance: mean(a->b) + mean(b->a), in mm."""
    d_ab = pcd_a.compute_point_cloud_distance(pcd_b)
    d_ba = pcd_b.compute_point_cloud_distance(pcd_a)
    return float(np.mean(d_ab) + np.mean(d_ba))


def prepare_optimization_data(noisy_dir, clean_dir,
                              noisy_suffix="_noisy", clean_suffix="_clean",
                              sample_points=DEFAULT_SAMPLE_POINTS):
    """Load Noise-Pair specimens and precompute their solidified volumes.

    Files are matched by replacing noisy_suffix with clean_suffix in the
    file stem (e.g. NP_MW_0001_noisy.stl <-> NP_MW_0001_clean.stl). For
    each pair, the noisy mesh is sampled, voxelized, and solidified once so
    that Optuna trials only need to re-run the erosion stage.
    """
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)

    noisy_files = sorted(noisy_dir.glob(f"*{noisy_suffix}.stl"))
    if not noisy_files:
        print(f"[Error] No '*{noisy_suffix}.stl' files found in {noisy_dir}.")
        return []

    dataset = []
    print("Loading pairs and precomputing solidified volumes...")
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

        voxelized = voxelize(pcd_noisy, voxel_res_factor=VOXEL_RES_FACTOR)
        if voxelized is None:
            print(f"[Warning] Degenerate geometry in {noisy_path.name}; "
                  "skipped.")
            continue
        shell_volume, min_index, voxel_size, origin = voxelized

        bbox = pcd_noisy.get_axis_aligned_bounding_box()
        extent = bbox.get_max_bound() - bbox.get_min_bound()

        closed_shell = ndimage.binary_dilation(
            shell_volume, structure=STRUCT_3D, iterations=SOLIDIFY_ITERATIONS
        )
        solid_volume = ndimage.binary_fill_holes(closed_shell)

        dataset.append({
            "solid_volume": solid_volume,
            "voxel_size": voxel_size,
            "object_max_size": float(np.max(extent)),
            "min_index": min_index,
            "origin": origin,
            "pcd_clean": pcd_clean,
        })

    print(f"Loaded {len(dataset)} pairs.")
    return dataset


# --- Objective -------------------------------------------------------------------

def objective(trial, dataset):
    """Mean Chamfer Distance after UE at the trial's removal ratio."""
    removal_ratio = trial.suggest_float(
        "removal_ratio", ALPHA_SEARCH_MIN, ALPHA_SEARCH_MAX
    )

    errors = []
    for data in dataset:
        erode_iter = compute_erosion_iterations(
            data["object_max_size"], data["voxel_size"],
            removal_ratio=removal_ratio,
        )

        eroded = ndimage.binary_erosion(
            data["solid_volume"], structure=STRUCT_3D, iterations=erode_iter
        )
        if not np.any(eroded):
            errors.append(EMPTY_RESULT_PENALTY)
            continue

        labeled, num_features = ndimage.label(eroded, structure=STRUCT_3D)
        sizes = ndimage.sum(eroded, labeled, range(num_features + 1))
        core_mask = (labeled == int(sizes[1:].argmax()) + 1)
        restored = ndimage.binary_dilation(
            core_mask, structure=STRUCT_3D, iterations=erode_iter
        )

        # Approximate the denoised geometry by the restored voxel centers
        # (geometry recovery is skipped for speed; see module docstring).
        occupied = np.argwhere(restored)
        centers = (occupied + data["min_index"]) * data["voxel_size"] \
            + data["origin"]
        pcd_result = o3d.geometry.PointCloud()
        pcd_result.points = o3d.utility.Vector3dVector(centers)

        errors.append(chamfer_distance(pcd_result, data["pcd_clean"]))

    return float(np.mean(errors))


# --- CLI ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize the UE removal ratio (alpha) via Optuna "
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
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the Optuna sampler (for reproducible "
             "optimization runs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = prepare_optimization_data(
        args.noisy_dir, args.clean_dir,
        noisy_suffix=args.noisy_suffix,
        clean_suffix=args.clean_suffix,
        sample_points=args.sample_points,
    )
    if not dataset:
        print("[Error] No valid pairs loaded; aborting.")
        return

    sampler = (optuna.samplers.TPESampler(seed=args.seed)
               if args.seed is not None else None)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: objective(trial, dataset),
                   n_trials=args.n_trials)

    best_ratio = study.best_params["removal_ratio"]
    print("\n" + "=" * 40)
    print("UE optimization result:")
    print(f"  Best removal_ratio (alpha): {best_ratio:.6f} "
          f"({best_ratio * 100:.3f}% of the object's max length)")
    print(f"  Minimum error             : {study.best_value:.6f}")
    print("=" * 40)


if __name__ == "__main__":
    main()