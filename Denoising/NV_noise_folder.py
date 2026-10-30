"""Extract noise patches from the Noise-Pair Dataset for NV training.

Implements step (1) of the Noise Verification data preparation (paper
Section 4.2.3): for each noisy/clean pair, points of the noisy specimen
farther than a threshold (0.5% of the object's maximum side length) from
the expert-cleaned ground truth are extracted as noise candidates,
separated into physically distinct clusters with DBSCAN, and each
sufficiently large cluster is saved as an individual PLY point cloud.
The resulting patches form the 'noise' class of the verifier's training
set (data_utils/noiseDataset.py expects them under noise_dataset_all/).

Pairing follows the released Noise-Pair naming (<stem>_noisy.<ext> /
<stem>_clean.<ext>); other suffix conventions can be given via
--noisy_suffix/--clean_suffix. Pairs whose files use different mesh
formats (e.g. noisy STL, clean PLY) are matched by stem.

Note on determinism: surface sampling from STL meshes is not seedable in
Open3D, so regenerated patches differ slightly between runs. The patch
set used in the paper is included in the released dataset; this script
documents how it was produced.

Example:
    python extract_noise_patches.py \
        --noisy_dir noise_pair/noisy --clean_dir noise_pair/clean \
        --output_dir data/noise_mini/noise_dataset_all
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

# --- Constants ---------------------------------------------------------------

# Points sampled from STL meshes (PLY point clouds are used as-is).
NUM_SAMPLE_POINTS = 100000

# Noise threshold as a fraction of the object's maximum side length
# (paper Section 4.2.3: 0.5%).
THRESHOLD_FACTOR = 0.005

# DBSCAN neighborhood radius = threshold distance x this factor.
DBSCAN_EPS_FACTOR = 1.5

# DBSCAN core-point neighbor count.
DBSCAN_MIN_POINTS = 10

# Minimum points for a cluster (and for the noise candidate set) to be kept.
MIN_CLUSTER_POINTS = 100


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


def find_clean_counterpart(noisy_path, clean_dir, noisy_suffix, clean_suffix):
    """Locate the expert-cleaned file paired with a noisy file.

    Tries the same extension first, then any extension with the same stem
    (pairs occasionally mix STL and PLY).
    """
    stem = noisy_path.stem
    if noisy_suffix and stem.endswith(noisy_suffix):
        stem = stem[: -len(noisy_suffix)]

    candidate = clean_dir / f"{stem}{clean_suffix}{noisy_path.suffix}"
    if candidate.exists():
        return candidate

    matches = sorted(clean_dir.glob(f"{stem}{clean_suffix}.*"))
    return matches[0] if matches else None


def extract_noise_patches(noisy_dir, clean_dir, output_dir,
                          noisy_suffix="_noisy", clean_suffix="_clean",
                          threshold_factor=THRESHOLD_FACTOR,
                          min_points=MIN_CLUSTER_POINTS,
                          sampling_points=NUM_SAMPLE_POINTS):
    """Extract per-cluster noise patches from every noisy/clean pair."""
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noisy_files = sorted(
        p for p in noisy_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".stl", ".ply"}
        and p.stem.endswith(noisy_suffix)
    )
    print(f"Total noisy files found: {len(noisy_files)}")

    total_parts = 0
    for noisy_path in noisy_files:
        clean_path = find_clean_counterpart(
            noisy_path, clean_dir, noisy_suffix, clean_suffix
        )
        if clean_path is None:
            print(f"Skipping: clean file not found for {noisy_path.name}")
            continue
        print(f"Pair found: [{noisy_path.name}] <-> [{clean_path.name}]")

        pcd_noisy = load_as_pcd(noisy_path, sampling_points)
        pcd_clean = load_as_pcd(clean_path, sampling_points)
        if pcd_noisy is None or pcd_clean is None:
            print("  [WARN] Failed to load pair; skipped.")
            continue

        # Noise candidates: points farther from the ground truth than
        # threshold_factor x (maximum bounding-box side length).
        dists = pcd_noisy.compute_point_cloud_distance(pcd_clean)
        bbox = pcd_noisy.get_axis_aligned_bounding_box()
        max_len = float(np.max(bbox.get_max_bound() - bbox.get_min_bound()))
        threshold = max_len * threshold_factor

        noise_indices = np.where(np.asarray(dists) > threshold)[0]
        if len(noise_indices) < min_points:
            print("  Too few noise points; skipped.")
            continue
        pcd_noise_only = pcd_noisy.select_by_index(noise_indices)

        # Separate physically distinct clusters.
        eps = threshold * DBSCAN_EPS_FACTOR
        labels = np.array(pcd_noise_only.cluster_dbscan(
            eps=eps, min_points=DBSCAN_MIN_POINTS))
        if labels.size == 0:
            continue

        stem = noisy_path.stem
        if noisy_suffix and stem.endswith(noisy_suffix):
            stem = stem[: -len(noisy_suffix)]

        count = 0
        for cluster_id in range(labels.max() + 1):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) < min_points:
                continue
            part_pcd = pcd_noise_only.select_by_index(indices)
            save_name = f"{stem}_noise_part_{cluster_id:03d}.ply"
            o3d.io.write_point_cloud(str(output_dir / save_name), part_pcd)
            count += 1

        total_parts += count
        print(f"  -> Extracted {count} noise parts.")

    print("-" * 40)
    print(f"Done. Total noise patches: {total_parts}")
    print(f"Output: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DBSCAN noise patches from Noise-Pair data "
                    "for Noise Verification training."
    )
    parser.add_argument("--noisy_dir", type=str, required=True,
                        help="Directory containing the noisy specimens.")
    parser.add_argument("--clean_dir", type=str, required=True,
                        help="Directory containing the expert-cleaned "
                             "specimens.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where noise patch PLYs are written "
                             "(noiseDataset.py expects noise_dataset_all/).")
    parser.add_argument("--noisy_suffix", type=str, default="_noisy",
                        help="Filename suffix of noisy files "
                             "(default: _noisy; legacy data used '(noise)').")
    parser.add_argument("--clean_suffix", type=str, default="_clean",
                        help="Filename suffix of cleaned files "
                             "(default: _clean; legacy data used "
                             "'(cleaned)').")
    parser.add_argument("--threshold_factor", type=float,
                        default=THRESHOLD_FACTOR,
                        help="Noise distance threshold as a fraction of the "
                             "object's maximum side length (default: 0.005).")
    parser.add_argument("--min_points", type=int, default=MIN_CLUSTER_POINTS,
                        help="Minimum points per kept cluster "
                             "(default: 100).")
    parser.add_argument("--sampling_points", type=int,
                        default=NUM_SAMPLE_POINTS,
                        help="Points sampled from STL meshes "
                             "(default: 100000).")
    return parser.parse_args()


def main():
    args = parse_args()
    extract_noise_patches(
        noisy_dir=args.noisy_dir,
        clean_dir=args.clean_dir,
        output_dir=args.output_dir,
        noisy_suffix=args.noisy_suffix,
        clean_suffix=args.clean_suffix,
        threshold_factor=args.threshold_factor,
        min_points=args.min_points,
        sampling_points=args.sampling_points,
    )


if __name__ == "__main__":
    main()