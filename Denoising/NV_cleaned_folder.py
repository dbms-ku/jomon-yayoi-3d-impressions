"""Generate positive (object) patches from cleaned data for NV training.

Implements step (2) of the Noise Verification data preparation (paper
Section 4.2.3): each expert-cleaned ground-truth specimen is deliberately
fragmented into multiple local surface patches with DBSCAN, so that the
verifier can learn the geometric character of genuine artifact surfaces.
Each sufficiently large patch is saved as an individual PLY point cloud.
The resulting patches form the 'positive' class of the verifier's
training set (data_utils/noiseDataset.py expects them under
positive_dataset_all/).

Parameter notes:
  * The DBSCAN radius is eps = max_side_length x --eps_factor. The
    experiments used 0.008 (the value passed at the original call site;
    the original function signature carried an unused default of 0.015).
    For reference, the noise-extraction script's effective radius is
    max_side_length x 0.005 x 1.5 = 0.0075, i.e. slightly SMALLER than
    the positive-side radius.
  * DBSCAN min_points is 20 here (10 in the noise-extraction script).
  * Patches are kept only if they contain at least --min_points points
    (default 100), matching the noise-side constraint.

Files are matched by the released Noise-Pair naming (<stem>_clean.<ext>);
other suffix conventions can be given via --clean_suffix.

Note on determinism: surface sampling from STL meshes is not seedable in
Open3D, so regenerated patches differ slightly between runs. The patch
set used in the paper is included in the released dataset; this script
documents how it was produced.

Example:
    python generate_positive_patches.py \
        --clean_dir noise_pair/clean \
        --output_dir data/noise_mini/positive_dataset_all
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

# --- Constants ---------------------------------------------------------------

# Points sampled from STL meshes (PLY point clouds are used as-is).
NUM_SAMPLE_POINTS = 100000

# DBSCAN neighborhood radius as a fraction of the object's maximum side
# length (value used for the paper's experiments).
EPS_FACTOR = 0.008

# DBSCAN core-point neighbor count.
DBSCAN_MIN_POINTS = 20

# Minimum points for a patch to be kept (matches the noise side).
MIN_PATCH_POINTS = 100


def load_as_pcd(file_path, sampling_points=NUM_SAMPLE_POINTS):
    """Load an STL (uniform surface sampling) or PLY file as a point cloud."""
    path = Path(file_path)
    if not path.exists():
        return None

    if path.suffix.lower() == ".stl":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.vertices) == 0:
            return None
        return mesh.sample_points_uniformly(number_of_points=sampling_points)
    return o3d.io.read_point_cloud(str(path))


def generate_positive_patches(clean_dir, output_dir,
                              clean_suffix="_clean",
                              eps_factor=EPS_FACTOR,
                              min_points=MIN_PATCH_POINTS,
                              sampling_points=NUM_SAMPLE_POINTS):
    """Fragment every cleaned specimen into local surface patches."""
    clean_dir = Path(clean_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(
        p for p in clean_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".stl", ".ply"}
        and p.stem.endswith(clean_suffix)
    )
    print(f"Total cleaned files found: {len(clean_files)}")

    total_parts = 0
    for clean_path in clean_files:
        print(f"Dividing: {clean_path.name}...")

        pcd = load_as_pcd(clean_path, sampling_points)
        if pcd is None:
            print("  [WARN] Failed to load; skipped.")
            continue

        # Fragment the ground-truth surface into local patches. The radius
        # controls how many pieces each specimen is divided into.
        bbox = pcd.get_axis_aligned_bounding_box()
        max_len = float(np.max(bbox.get_max_bound() - bbox.get_min_bound()))
        eps = max_len * eps_factor
        labels = np.array(pcd.cluster_dbscan(eps=eps,
                                             min_points=DBSCAN_MIN_POINTS))
        if labels.size == 0:
            continue

        stem = clean_path.stem
        if clean_suffix and stem.endswith(clean_suffix):
            stem = stem[: -len(clean_suffix)]

        count = 0
        for cluster_id in range(labels.max() + 1):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) < min_points:
                continue
            part_pcd = pcd.select_by_index(indices)
            save_name = f"{stem}_pos_part_{cluster_id:03d}.ply"
            o3d.io.write_point_cloud(str(output_dir / save_name), part_pcd)
            count += 1

        total_parts += count
        print(f"  -> Generated {count} positive parts.")

    print("-" * 40)
    print(f"Done. Total positive patches: {total_parts}")
    print(f"Output: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fragment cleaned Noise-Pair specimens into positive "
                    "(object) patches for Noise Verification training."
    )
    parser.add_argument("--clean_dir", type=str, required=True,
                        help="Directory containing the expert-cleaned "
                             "specimens.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where positive patch PLYs are "
                             "written (noiseDataset.py expects "
                             "positive_dataset_all/).")
    parser.add_argument("--clean_suffix", type=str, default="_clean",
                        help="Filename suffix of cleaned files "
                             "(default: _clean; legacy data used "
                             "'(cleaned)').")
    parser.add_argument("--eps_factor", type=float, default=EPS_FACTOR,
                        help="DBSCAN radius as a fraction of the object's "
                             "maximum side length (default: 0.008, the "
                             "paper's value).")
    parser.add_argument("--min_points", type=int, default=MIN_PATCH_POINTS,
                        help="Minimum points per kept patch (default: 100).")
    parser.add_argument("--sampling_points", type=int,
                        default=NUM_SAMPLE_POINTS,
                        help="Points sampled from STL meshes "
                             "(default: 100000).")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_positive_patches(
        clean_dir=args.clean_dir,
        output_dir=args.output_dir,
        clean_suffix=args.clean_suffix,
        eps_factor=args.eps_factor,
        min_points=args.min_points,
        sampling_points=args.sampling_points,
    )


if __name__ == "__main__":
    main()