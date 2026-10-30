"""Adaptive Neck Removal (ANR) for 3D pottery-impression point clouds.

This script implements the Adaptive Neck Removal method described in the
accompanying paper (Section 4.2.2). Thin "neck" structures that connect
noise attachments to the main impression body are identified via the
Euclidean Distance Transform (EDT) in voxel space and disconnected, after
which only the primary structures are retained. The denoised voxel mask is
finally used to select the corresponding points from the original
high-resolution point cloud.

Input meshes/point clouds are read from a directory tree and the denoised
results are written to a mirrored directory tree.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage

# --- Constants ---------------------------------------------------------------

# Number of points sampled from each input mesh (Poisson-disk sampling).
NUM_SAMPLE_POINTS = 50000

# Voxel edge length as a fraction of the object's longest bounding-box side
# (V_s = L_max * VOXEL_RES_FACTOR in the paper).
VOXEL_RES_FACTOR = 0.005

# A survivor voxel center and an original point are treated as coincident when
# their distance is below (voxel_size * VOXEL_MATCH_FACTOR). The factor is
# slightly below sqrt(3)/2 (~0.866), i.e. half the voxel diagonal, so that any
# original point falling inside a survivor voxel is recovered.
VOXEL_MATCH_FACTOR = 0.87

# Supported input geometry formats.
INPUT_EXTENSIONS = (".stl", ".ply", ".obj")

# 6-connectivity structuring element for 3D morphological operations.
STRUCT_3D = ndimage.generate_binary_structure(3, 1)

# Default number of binary-dilation iterations applied before hole filling
# during solidification (0 disables the pre-dilation).
SOLIDIFY_ITERATIONS = 2


# --- I/O ---------------------------------------------------------------------

def load_as_point_cloud(path):
    """Load a mesh or point cloud file and return an Open3D point cloud.

    If the file is a mesh with faces, points are sampled from its surface via
    Poisson-disk sampling. Otherwise the file is read directly as a point
    cloud. Returns None if nothing usable could be loaded.
    """
    abs_path = str(Path(path).resolve())

    try:
        mesh = o3d.io.read_triangle_mesh(abs_path)
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            return mesh.sample_points_poisson_disk(NUM_SAMPLE_POINTS)
    except Exception:
        # Fall back to reading the file as a raw point cloud below.
        pass

    pcd = o3d.io.read_point_cloud(abs_path)
    if pcd is None or len(pcd.points) == 0:
        return None
    return pcd


# --- Core denoising ----------------------------------------------------------

def voxelize(pcd, voxel_res_factor=VOXEL_RES_FACTOR):
    """Voxelize a point cloud with an adaptive, size-dependent voxel size.

    Returns a tuple (shell_volume, min_index, voxel_size, origin) where
    shell_volume is a boolean 3D array marking occupied voxels. Returns None
    if the point cloud is degenerate (e.g. zero extent or no voxels).
    """
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    max_len = float(np.max(extent))
    if max_len <= 0.0:
        return None

    voxel_size = max_len * voxel_res_factor
    if voxel_size <= 0.0:
        return None

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size
    )
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        return None

    indices = np.stack([v.grid_index for v in voxels])
    min_index = indices.min(axis=0)
    dims = indices.max(axis=0) - min_index + 1

    shell_volume = np.zeros(dims, dtype=bool)
    local = indices - min_index
    shell_volume[local[:, 0], local[:, 1], local[:, 2]] = True

    return shell_volume, min_index, voxel_size, voxel_grid.origin


def compute_survivor_mask(solid_volume, break_ratio, survival_ratio):
    """Identify primary structures via EDT-based neck removal.

    For each connected component, the EDT gives the local thickness. Voxels
    thinner than (max_radius * break_ratio) are treated as necks and removed,
    splitting the component into parts. A part survives if its maximum local
    thickness exceeds (max_radius * survival_ratio); survivors are dilated back
    to compensate for the removed necks.
    """
    labeled, num_features = ndimage.label(solid_volume, structure=STRUCT_3D)
    survivor_mask = np.zeros_like(solid_volume, dtype=bool)

    for label_id in range(1, num_features + 1):
        component_mask = (labeled == label_id)
        edt = ndimage.distance_transform_edt(component_mask)
        max_radius = edt.max()
        threshold = max_radius * break_ratio

        # Voxels thicker than the neck threshold form the component's "core".
        core_mask = (edt > threshold)
        if not np.any(core_mask):
            # No neck could be cut; keep the whole component.
            survivor_mask |= component_mask
            continue

        dilate_iter = int(threshold)
        sub_labeled, sub_num = ndimage.label(core_mask, structure=STRUCT_3D)
        for sub_id in range(1, sub_num + 1):
            sub_mask = (sub_labeled == sub_id)
            part_max_thickness = np.max(edt[sub_mask])
            if part_max_thickness > (max_radius * survival_ratio):
                restored = ndimage.binary_dilation(
                    sub_mask, structure=STRUCT_3D, iterations=dilate_iter
                )
                survivor_mask |= restored

    return survivor_mask


def denoise_point_cloud(pcd, break_ratio, survival_ratio,
                        solidify_iterations=2,
                        voxel_res_factor=VOXEL_RES_FACTOR):
    """Run the full ANR pipeline on a point cloud and return the denoised one.

    Returns None if the point cloud cannot be processed.

    Note:
        solidify_iterations controls a binary dilation applied before hole
        filling to close small surface gaps prior to the volumetric
        operations. Set to 0 to fill holes without any pre-dilation.
    """
    voxelized = voxelize(pcd, voxel_res_factor=voxel_res_factor)
    if voxelized is None:
        return None
    shell_volume, min_index, voxel_size, origin = voxelized

    # Solidification: close gaps, then fill interior holes to obtain a solid.
    if solidify_iterations > 0:
        shell_volume = ndimage.binary_dilation(
            shell_volume, structure=STRUCT_3D, iterations=solidify_iterations
        )
    solid_volume = ndimage.binary_fill_holes(shell_volume)

    survivor_mask = compute_survivor_mask(
        solid_volume, break_ratio, survival_ratio
    )

    occupied = np.argwhere(survivor_mask)  # rows of (i0, i1, i2) grid indices
    if occupied.shape[0] == 0:
        return None

    # Map surviving voxels back to world coordinates and recover the
    # corresponding high-resolution points from the original cloud.
    survivor_centers = (occupied + min_index) * voxel_size + origin
    survivor_pcd = o3d.geometry.PointCloud()
    survivor_pcd.points = o3d.utility.Vector3dVector(survivor_centers)

    dists = np.asarray(pcd.compute_point_cloud_distance(survivor_pcd))
    keep_indices = np.where(dists < voxel_size * VOXEL_MATCH_FACTOR)[0]
    if keep_indices.size == 0:
        return None

    return pcd.select_by_index(keep_indices)


def process_file(input_path, output_path, break_ratio, survival_ratio,
                 solidify_iterations=2):
    """Denoise a single file and write the result. Returns True on success."""
    pcd = load_as_point_cloud(input_path)
    if pcd is None:
        return False

    denoised = denoise_point_cloud(
        pcd, break_ratio, survival_ratio,
        solidify_iterations=solidify_iterations,
    )
    if denoised is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path.resolve()), denoised)
    return True


# --- CLI ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive Neck Removal (ANR) denoising for 3D impressions."
    )
    parser.add_argument(
        "--break_ratio", type=float, default=0.20,
        help="ANR break ratio (beta): neck threshold as a fraction of the "
             "component's maximum local thickness (default: 0.20).",
    )
    parser.add_argument(
        "--survival_ratio", type=float, default=0.10,
        help="ANR survival ratio (gamma): minimum part thickness, as a "
             "fraction of the maximum, required to survive (default: 0.10).",
    )
    parser.add_argument(
        "--solidify_iterations", type=int, default=2,
        help="Dilation iterations applied before hole filling during "
             "solidification (default: 2; set 0 to disable).",
    )
    parser.add_argument(
        "--input_root", type=str, default="./data/input",
        help="Root directory containing input meshes/point clouds.",
    )
    parser.add_argument(
        "--output_root", type=str, default="./data/output_anr",
        help="Root directory where denoised point clouds are written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    print("Starting Adaptive Neck Removal (ANR) denoising.")
    print(f"  Input root : {input_root}")
    print(f"  Output root: {output_root}")
    print(f"  break_ratio={args.break_ratio}, "
          f"survival_ratio={args.survival_ratio}, "
          f"solidify_iterations={args.solidify_iterations}")

    total = 0
    succeeded = 0
    for root, _dirs, files in os.walk(str(input_root)):
        for filename in files:
            if not filename.lower().endswith(INPUT_EXTENSIONS):
                continue

            total += 1
            input_path = Path(root) / filename
            relative_path = input_path.relative_to(input_root)
            output_path = (output_root / relative_path).with_suffix(".ply")

            print(f"[{total}] Processing: {relative_path}")
            try:
                if process_file(
                    input_path, output_path,
                    args.break_ratio, args.survival_ratio,
                    solidify_iterations=args.solidify_iterations,
                ):
                    succeeded += 1
                else:
                    print(f"  [Skipped] No usable geometry: {filename}")
            except Exception as exc:  # noqa: BLE001 - keep batch running
                print(f"  [Error] Failed: {filename} -> {exc}")

    print("-" * 40)
    print("Completed.")
    print(f"  Total files found : {total}")
    print(f"  Successfully saved: {succeeded}")


if __name__ == "__main__":
    main()