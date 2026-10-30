"""Adaptive Neck Removal with PointNet-based Noise Verification (ANR + NV).

This script implements the combined denoising pipeline described in the
accompanying paper (Sections 4.2.2 and 4.2.3), corresponding to the
best-performing configuration "SA-PointNet++ ANR-Noise Verification". As
in plain ANR, each connected component is segmented into sub-parts by
cutting thin "necks" identified via the Euclidean Distance Transform.
Unlike plain ANR, every sub-part is then judged by a trained PointNet
binary classifier, and a part is retained if EITHER of the following
holds:

  * geometric survival: its maximum local thickness exceeds
    (max_radius * survival_ratio), or
  * learned verification: the PointNet probability of the "positive"
    (object) class is at least --ai_threshold.

The verification step therefore rescues object parts that the geometric
rule alone would misidentify as noise.

Class convention: the verifier is trained on two classes named "noise"
and "positive"; with alphabetical index assignment this yields
index 0 = noise and index 1 = positive (object). The default
--positive_class_index is 1 accordingly.

Note on resolution: for efficiency, this combined pipeline caps the voxel
grid at --grid_limit cells along the longest side (default 128, i.e. a
voxel size of L_max/128 ~ L_max * 0.0078), which is coarser than the
L_max * 0.005 resolution used by the standalone UE/ANR scripts.

External dependency: the PointNet model definition (pointnet_cls.py) and
checkpoint layout follow the widely used PyTorch PointNet/PointNet++
implementation; point --models_root at its `models` directory and
--model_dir at the training log directory containing
checkpoints/best_model.pth.
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage
import torch
from tqdm import tqdm

from ANR_denoise import (
    SOLIDIFY_ITERATIONS,
    STRUCT_3D,
    VOXEL_MATCH_FACTOR,
    VOXEL_RES_FACTOR,
    load_as_point_cloud,
    voxelize,
)

# --- Constants ---------------------------------------------------------------

# Sub-parts smaller than this many voxels are discarded without judging.
MIN_PART_VOXELS = 10

# Recovered parts with fewer points than this are discarded without judging.
MIN_PART_POINTS = 20

# If the final result has fewer points than this, the original point cloud
# is written unmodified as a safe fallback.
MIN_RESULT_POINTS = 50


# --- PointNet preprocessing ----------------------------------------------------

def pc_normalize(points):
    """Center a point set and scale it into the unit sphere."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    return points / scale


def farthest_point_sample(point, npoint):
    """Subsample a point set to npoint points via Farthest Point Sampling."""
    N, _ = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return point[centroids.astype(np.int32)]


def adjust_point_count(points, num_point):
    """Bring a point set to exactly num_point points.

    Mirrors the training-time preprocessing of the verifier
    (data_utils/noiseDataset.py): Farthest Point Sampling when oversized,
    random duplication when undersized.
    """
    if len(points) >= num_point:
        return farthest_point_sample(points, num_point)
    idx = np.random.choice(len(points), num_point, replace=True)
    return points[idx]


def load_verifier(model_dir, models_root, device):
    """Load the trained PointNet noise/positive binary classifier."""
    models_root = Path(models_root)
    if not models_root.exists():
        raise FileNotFoundError(
            f"PointNet model definitions not found: {models_root}"
        )
    sys.path.append(str(models_root.resolve()))
    module = importlib.import_module("pointnet_cls")

    classifier = module.get_model(2, normal_channel=False).to(device)
    checkpoint_path = Path(model_dir) / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    return classifier


def predict_positive_probability(classifier, points, num_point,
                                 positive_class_index, device):
    """Return the verifier's probability that a part is object geometry."""
    pts = adjust_point_count(points.astype(np.float32), num_point)
    pts = pc_normalize(pts)
    tensor = torch.Tensor(pts).unsqueeze(0).transpose(2, 1).to(device)
    with torch.no_grad():
        pred, _ = classifier(tensor)
        prob = torch.nn.functional.softmax(pred, dim=1)
    return float(prob[0][positive_class_index].item())


# --- Core pipeline ---------------------------------------------------------------

def denoise_with_verification(pcd, classifier, break_ratio, survival_ratio,
                              ai_threshold, num_point,
                              positive_class_index, device,
                              grid_limit=128):
    """Run ANR segmentation with PointNet verification on a point cloud.

    Returns a tuple (denoised_pcd, status) where status is one of:
      'ok'       - denoising succeeded.
      'fallback' - too few points survived; the original cloud is returned.
      'failed'   - the point cloud could not be processed.
    """
    # Cap the voxel resolution at grid_limit cells along the longest side
    # (see module docstring); this is coarser than the standalone scripts.
    effective_factor = max(1.0 / grid_limit, VOXEL_RES_FACTOR)
    voxelized = voxelize(pcd, voxel_res_factor=effective_factor)
    if voxelized is None:
        return None, "failed"
    shell_volume, min_index, voxel_size, origin = voxelized

    closed_shell = ndimage.binary_dilation(
        shell_volume, structure=STRUCT_3D, iterations=SOLIDIFY_ITERATIONS
    )
    solid_volume = ndimage.binary_fill_holes(closed_shell)

    labeled, num_features = ndimage.label(solid_volume, structure=STRUCT_3D)
    final_pcd = o3d.geometry.PointCloud()

    for label_id in range(1, num_features + 1):
        component_mask = (labeled == label_id)
        edt = ndimage.distance_transform_edt(component_mask)
        max_radius = edt.max()
        cut_threshold = max_radius * break_ratio
        core_mask = (edt > cut_threshold)

        if not np.any(core_mask):
            # No neck could be cut; judge the whole component as one part.
            sub_parts = [component_mask]
            dilate_iter = 0
        else:
            sub_labeled, sub_num = ndimage.label(core_mask,
                                                 structure=STRUCT_3D)
            sub_parts = [(sub_labeled == sid)
                         for sid in range(1, sub_num + 1)]
            dilate_iter = int(cut_threshold)

        for part_mask in sub_parts:
            if dilate_iter > 0:
                restored = ndimage.binary_dilation(
                    part_mask, structure=STRUCT_3D, iterations=dilate_iter
                )
            else:
                restored = part_mask

            occupied = np.argwhere(restored)
            if occupied.shape[0] < MIN_PART_VOXELS:
                continue

            # Recover the original high-resolution points of this part.
            centers = (occupied + min_index) * voxel_size + origin
            part_voxels = o3d.geometry.PointCloud()
            part_voxels.points = o3d.utility.Vector3dVector(centers)

            dists = np.asarray(pcd.compute_point_cloud_distance(part_voxels))
            part_indices = np.where(
                dists < voxel_size * VOXEL_MATCH_FACTOR
            )[0]
            part_pcd = pcd.select_by_index(part_indices)
            if len(part_pcd.points) < MIN_PART_POINTS:
                continue

            # Keep the part if it survives geometrically OR the verifier
            # judges it to be object geometry.
            part_max_thickness = float(np.max(edt[part_mask]))
            survives_geometrically = (
                part_max_thickness > (max_radius * survival_ratio)
            )
            if survives_geometrically:
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
        description="ANR denoising with PointNet-based Noise Verification."
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
        "--break_ratio", type=float, default=0.20,
        help="ANR break ratio (beta) used to cut necks (default: 0.20).",
    )
    parser.add_argument(
        "--survival_ratio", type=float, default=0.10,
        help="ANR survival ratio (gamma) for the geometric criterion "
             "(default: 0.10).",
    )
    parser.add_argument(
        "--ai_threshold", type=float, default=0.05,
        help="Minimum 'positive' probability for the verifier to rescue a "
             "part. Deliberately low so the pipeline errs on the side of "
             "keeping geometry (default: 0.05).",
    )
    parser.add_argument(
        "--positive_class_index", type=int, default=1,
        help="Output index of the 'positive' (object) class in the "
             "verifier. With classes named 'noise'/'positive' and "
             "alphabetical assignment this is 1 (default: 1).",
    )
    parser.add_argument(
        "--num_point", type=int, default=256,
        help="Points fed to the verifier per part (default: 256).",
    )
    parser.add_argument(
        "--grid_limit", type=int, default=128,
        help="Maximum voxel grid cells along the longest side "
             "(default: 128).",
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
        if p.is_file() and p.suffix.lower() in {".stl", ".ply", ".obj"}
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
                break_ratio=args.break_ratio,
                survival_ratio=args.survival_ratio,
                ai_threshold=args.ai_threshold,
                num_point=args.num_point,
                positive_class_index=args.positive_class_index,
                device=device,
                grid_limit=args.grid_limit,
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