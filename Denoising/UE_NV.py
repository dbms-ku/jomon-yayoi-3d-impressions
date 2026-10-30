"""Uniform Erosion with PointNet-based Noise Verification (UE + NV).

This script implements the combined denoising pipeline described in the
accompanying paper (Sections 4.2.1 and 4.2.3), corresponding to the
configuration "SA-PointNet++ UE-Noise Verification". A morphological
opening (erosion -> dilation) in voxel space disconnects thin attachments,
after which every resulting connected component is judged:

  * the largest component (the main impression body) is always kept, and
  * every other component is kept if the PointNet verifier's probability
    of the "positive" (object) class is at least --ai_threshold.

The verification step therefore rescues object parts that plain UE (which
keeps only the largest component) would discard.

Class convention: the verifier is trained on two classes named "noise"
and "positive"; with alphabetical index assignment this yields
index 0 = noise and index 1 = positive (object).

Note on implementation details: this combined pipeline intentionally
mirrors the script used for the paper's experiments and differs from the
standalone UE script in several particulars (uniform sampling of 100,000
points instead of Poisson-disk 50,000; one solidification dilation
instead of two; no +1 safety margin on the erosion iterations; a voxel
match factor of 0.9 instead of 0.87). These differences are documented in
the repository README.

External dependency: the PointNet model definition (pointnet_cls.py) and
checkpoint layout follow the widely used PyTorch PointNet/PointNet++
implementation; see anr_nv_denoise.py for details.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage
import torch
from tqdm import tqdm

from UE_denoise import STRUCT_3D, voxelize
from ANR_NV import load_verifier, predict_positive_probability

# --- Constants ---------------------------------------------------------------

# Number of points uniformly sampled from each input mesh.
NUM_SAMPLE_POINTS = 100000

# Voxel edge length as a fraction of the object's longest bounding-box side.
VOXEL_RES_FACTOR = 0.005

# A kept voxel center and an original point are treated as coincident when
# their distance is below (voxel_size * VOXEL_MATCH_FACTOR).
VOXEL_MATCH_FACTOR = 0.9

# Dilation iterations applied to the voxel shell before hole filling.
SOLIDIFY_ITERATIONS = 1

# Components smaller than this many voxels are discarded without judging.
MIN_PART_VOXELS = 10

# Recovered parts with fewer points than this are discarded without judging.
MIN_PART_POINTS = 20

# If the final result has fewer points than this, the original point cloud
# is written unmodified as a safe fallback.
MIN_RESULT_POINTS = 50


# --- I/O ---------------------------------------------------------------------

def load_as_point_cloud(path, num_points=NUM_SAMPLE_POINTS):
    """Load a mesh (uniform surface sampling) or a point cloud file.

    Returns None if nothing usable could be loaded.
    """
    path = Path(path)
    if path.suffix.lower() == ".stl":
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return None
        return mesh.sample_points_uniformly(number_of_points=num_points)

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd is None or len(pcd.points) < 10:
        return None
    return pcd


# --- Core pipeline ---------------------------------------------------------------

def denoise_with_verification(pcd, classifier, removal_ratio, ai_threshold,
                              num_point, positive_class_index, device,
                              voxel_res_factor=VOXEL_RES_FACTOR):
    """Run UE opening with PointNet verification on a point cloud.

    Returns a tuple (denoised_pcd, status) where status is one of:
      'ok'       - denoising succeeded.
      'fallback' - too few points survived; the original cloud is returned.
      'failed'   - the point cloud could not be processed.
    """
    voxelized = voxelize(pcd, voxel_res_factor=voxel_res_factor)
    if voxelized is None:
        return None, "failed"
    shell_volume, min_index, voxel_size, origin = voxelized

    bbox = pcd.get_axis_aligned_bounding_box()
    object_max_size = float(np.max(bbox.get_max_bound() - bbox.get_min_bound()))

    # Solidification: close small gaps, then fill interior holes.
    closed_shell = ndimage.binary_dilation(
        shell_volume, structure=STRUCT_3D, iterations=SOLIDIFY_ITERATIONS
    )
    solid_volume = ndimage.binary_fill_holes(closed_shell)

    # Opening: erosion followed by dilation disconnects thin attachments.
    target_thickness = object_max_size * removal_ratio
    erode_iter = int(np.ceil((target_thickness / 2.0) / voxel_size))
    if erode_iter < 1:
        erode_iter = 1

    eroded = ndimage.binary_erosion(
        solid_volume, structure=STRUCT_3D, iterations=erode_iter
    )
    if not np.any(eroded):
        # The erosion depth exceeded the object's thickness.
        return pcd, "fallback"
    restored = ndimage.binary_dilation(
        eroded, structure=STRUCT_3D, iterations=erode_iter
    )

    # Label the disconnected components and identify the main body once.
    labeled, num_features = ndimage.label(restored, structure=STRUCT_3D)
    sizes = ndimage.sum(restored, labeled, range(num_features + 1))
    largest_label = int(sizes[1:].argmax()) + 1

    final_pcd = o3d.geometry.PointCloud()
    for label_id in range(1, num_features + 1):
        occupied = np.argwhere(labeled == label_id)
        if occupied.shape[0] < MIN_PART_VOXELS:
            continue

        # Recover the original high-resolution points of this component.
        centers = (occupied + min_index) * voxel_size + origin
        part_voxels = o3d.geometry.PointCloud()
        part_voxels.points = o3d.utility.Vector3dVector(centers)

        dists = np.asarray(pcd.compute_point_cloud_distance(part_voxels))
        part_indices = np.where(dists < voxel_size * VOXEL_MATCH_FACTOR)[0]
        part_pcd = pcd.select_by_index(part_indices)
        if len(part_pcd.points) < MIN_PART_POINTS:
            continue

        # Keep the main body unconditionally; rescue other components when
        # the verifier judges them to be object geometry.
        if label_id == largest_label:
            final_pcd += part_pcd
            continue

        positive_prob = predict_positive_probability(
            classifier, np.asarray(part_pcd.points),
            num_point, positive_class_index, device,
        )
        if positive_prob >= ai_threshold:
            final_pcd += part_pcd

    if len(final_pcd.points) < MIN_RESULT_POINTS:
        return pcd, "fallback"
    return final_pcd, "ok"


# --- CLI ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="UE denoising with PointNet-based Noise Verification."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Root directory containing input meshes/point clouds.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Root directory where denoised point clouds are written.",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Training log directory of the noise-verification PointNet "
             "(must contain checkpoints/best_model.pth).",
    )
    parser.add_argument(
        "--models_root", type=str, default="./models",
        help="Directory containing pointnet_cls.py (default: ./models).",
    )
    parser.add_argument(
        "--removal_ratio", type=float, default=0.01,
        help="UE removal ratio (alpha): target neck thickness as a "
             "fraction of the object's maximum side length (default: 0.01, "
             "the paper's optimized value).",
    )
    parser.add_argument(
        "--ai_threshold", type=float, default=0.05,
        help="Minimum 'positive' probability for the verifier to rescue a "
             "component. Deliberately low so the pipeline errs on the side "
             "of keeping geometry (default: 0.05).",
    )
    parser.add_argument(
        "--positive_class_index", type=int, default=1,
        help="Output index of the 'positive' (object) class in the "
             "verifier (default: 1).",
    )
    parser.add_argument(
        "--num_point", type=int, default=256,
        help="Points fed to the verifier per component (default: 256).",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Computation device (default: cuda if available, else cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    classifier = load_verifier(args.model_dir, args.models_root, device)

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    files = sorted(
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".stl", ".ply"}
    )
    print(f"[INFO] Files found: {len(files)}")

    counts = {"ok": 0, "fallback": 0, "failed": 0}
    for file_path in tqdm(files, desc="Processing"):
        try:
            pcd = load_as_point_cloud(file_path)
            if pcd is None:
                counts["failed"] += 1
                continue

            denoised, status = denoise_with_verification(
                pcd, classifier,
                removal_ratio=args.removal_ratio,
                ai_threshold=args.ai_threshold,
                num_point=args.num_point,
                positive_class_index=args.positive_class_index,
                device=device,
            )
            if denoised is None:
                counts["failed"] += 1
                continue

            relative_path = file_path.relative_to(input_root)
            output_file = (output_root / relative_path).with_suffix(".ply")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(output_file.resolve()), denoised)
            counts[status] += 1
        except Exception as exc:  # noqa: BLE001 - keep batch running
            print(f"\n[Error] Failed: {file_path.name} -> {exc}")
            counts["failed"] += 1

    print("-" * 40)
    print("Completed.")
    print(f"  Denoised            : {counts['ok']}")
    print(f"  Fallback (too small): {counts['fallback']}")
    print(f"  Failed              : {counts['failed']}")
    print(f"[INFO] Results saved to: {output_root}")


if __name__ == "__main__":
    main()