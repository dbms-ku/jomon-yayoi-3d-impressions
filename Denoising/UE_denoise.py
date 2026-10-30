"""Uniform Erosion (UE) denoising for 3D pottery-impression point clouds.

This script implements the Uniform Erosion method described in the
accompanying paper (Section 4.2.1). A morphological opening operation in
voxel space (erosion -> largest-connected-component extraction -> dilation)
removes thin attachments and isolated noise, after which the denoised voxel
mask is used to select the corresponding points from the original
high-resolution point cloud.

The erosion depth is controlled by the removal ratio alpha (paper Section
4.2.1): structures thinner than (alpha * L_max), where L_max is the
object's maximum bounding-box side length, are broken off. The
Bayesian-optimized value used in the paper is alpha = 0.01. An absolute
specification in millimetres (--neck_thickness_mm) is also available.

By default the denoised point cloud is written as-is; --output_mesh
additionally re-meshes it via alpha-shape reconstruction for mesh-based
downstream pipelines.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.ndimage as ndimage
from tqdm import tqdm

# --- Constants ---------------------------------------------------------------

# Number of points sampled from each input mesh (Poisson-disk sampling).
DEFAULT_NUM_SAMPLE_POINTS = 50000

# Voxel edge length as a fraction of the object's longest bounding-box side
# (V_s = L_max * VOXEL_RES_FACTOR in the paper).
VOXEL_RES_FACTOR = 0.005

# Bayesian-optimized removal ratio reported in the paper (alpha* = 0.01).
DEFAULT_REMOVAL_RATIO = 0.01

# A kept voxel center and an original point are treated as coincident when
# their distance is below (voxel_size * VOXEL_MATCH_FACTOR). The factor is
# slightly below sqrt(3)/2 (~0.866), i.e. half the voxel diagonal, so that any
# original point falling inside a kept voxel is recovered.
VOXEL_MATCH_FACTOR = 0.87

# Alpha-shape radius, as a multiple of the voxel size, used when re-meshing
# the denoised point cloud (--output_mesh).
ALPHA_SHAPE_FACTOR = 4.0

# Dilation iterations applied to the voxel shell before hole filling
# (closes small surface gaps so the interior can be filled reliably).
SOLIDIFY_ITERATIONS = 2

# Supported input geometry formats.
INPUT_EXTENSIONS = {".stl", ".ply", ".obj", ".off"}

# 6-connectivity structuring element for 3D morphological operations.
STRUCT_3D = ndimage.generate_binary_structure(3, 1)


# --- I/O ---------------------------------------------------------------------

def load_as_point_cloud(path, num_points=DEFAULT_NUM_SAMPLE_POINTS):
    """Load a mesh or point cloud file and return an Open3D point cloud.

    If the file is a mesh with faces, points are sampled from its surface via
    Poisson-disk sampling. Otherwise the file is read directly as a point
    cloud. Returns None if nothing usable could be loaded.
    """
    abs_path = str(Path(path).resolve())

    try:
        mesh = o3d.io.read_triangle_mesh(abs_path)
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            return mesh.sample_points_poisson_disk(number_of_points=num_points)
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


def compute_erosion_iterations(object_max_size, voxel_size,
                               removal_ratio=None, neck_thickness_mm=None):
    """Convert the requested erosion depth into a voxel iteration count.

    Exactly one of removal_ratio (paper's alpha, relative to the object's
    maximum side length) or neck_thickness_mm (absolute) must be given. The
    target thickness is halved to obtain the erosion radius, and one extra
    iteration is added as a safety margin so that necks at exactly the
    target thickness are reliably broken.
    """
    if (removal_ratio is None) == (neck_thickness_mm is None):
        raise ValueError(
            "Specify exactly one of removal_ratio or neck_thickness_mm."
        )

    if removal_ratio is not None:
        target_thickness = object_max_size * removal_ratio
    else:
        target_thickness = neck_thickness_mm

    erosion_radius = target_thickness / 2.0
    return int(np.ceil(erosion_radius / voxel_size)) + 1


def denoise_point_cloud(pcd, removal_ratio=None, neck_thickness_mm=None,
                        voxel_res_factor=VOXEL_RES_FACTOR):
    """Run the full UE pipeline on a point cloud.

    Returns a tuple (denoised_pcd, voxel_size, status) where status is one of:
      'ok'       - denoising succeeded.
      'vanished' - erosion removed everything; the original cloud is
                   returned unmodified as a safe fallback.
      'failed'   - the point cloud could not be processed (denoised_pcd
                   is None).
    """
    voxelized = voxelize(pcd, voxel_res_factor=voxel_res_factor)
    if voxelized is None:
        return None, None, "failed"
    shell_volume, min_index, voxel_size, origin = voxelized

    bbox = pcd.get_axis_aligned_bounding_box()
    object_max_size = float(np.max(bbox.get_max_bound() - bbox.get_min_bound()))

    erode_iter = compute_erosion_iterations(
        object_max_size, voxel_size,
        removal_ratio=removal_ratio,
        neck_thickness_mm=neck_thickness_mm,
    )

    # Solidification: close small gaps, then fill interior holes.
    closed_shell = ndimage.binary_dilation(
        shell_volume, structure=STRUCT_3D, iterations=SOLIDIFY_ITERATIONS
    )
    solid_volume = ndimage.binary_fill_holes(closed_shell)

    # Opening: erosion -> largest connected component -> dilation.
    eroded = ndimage.binary_erosion(
        solid_volume, structure=STRUCT_3D, iterations=erode_iter
    )

    if not np.any(eroded):
        # The erosion depth exceeded the object's thickness. Fall back to
        # the original point cloud rather than emitting an empty result.
        return pcd, voxel_size, "vanished"

    labeled, num_features = ndimage.label(eroded, structure=STRUCT_3D)
    sizes = ndimage.sum(eroded, labeled, range(num_features + 1))
    largest_label = int(sizes[1:].argmax()) + 1
    core_mask = (labeled == largest_label)

    restored = ndimage.binary_dilation(
        core_mask, structure=STRUCT_3D, iterations=erode_iter
    )

    # Map kept voxels back to world coordinates and recover the
    # corresponding high-resolution points from the original cloud.
    occupied = np.argwhere(restored)  # rows of (i0, i1, i2) grid indices
    kept_centers = (occupied + min_index) * voxel_size + origin

    kept_pcd = o3d.geometry.PointCloud()
    kept_pcd.points = o3d.utility.Vector3dVector(kept_centers)

    dists = np.asarray(pcd.compute_point_cloud_distance(kept_pcd))
    keep_indices = np.where(dists < voxel_size * VOXEL_MATCH_FACTOR)[0]
    if keep_indices.size == 0:
        return None, voxel_size, "failed"

    return pcd.select_by_index(keep_indices), voxel_size, "ok"


def reconstruct_mesh(pcd, voxel_size, alpha_factor=ALPHA_SHAPE_FACTOR):
    """Re-mesh a denoised point cloud via alpha-shape reconstruction."""
    alpha = voxel_size * alpha_factor
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )
    mesh.compute_vertex_normals()
    return mesh


def process_file(input_path, output_path, removal_ratio=None,
                 neck_thickness_mm=None,
                 num_points=DEFAULT_NUM_SAMPLE_POINTS,
                 output_mesh=False, visualize=False):
    """Denoise a single file and write the result. Returns the status string."""
    pcd = load_as_point_cloud(input_path, num_points=num_points)
    if pcd is None:
        return "failed"

    denoised, voxel_size, status = denoise_point_cloud(
        pcd,
        removal_ratio=removal_ratio,
        neck_thickness_mm=neck_thickness_mm,
    )
    if denoised is None:
        return "failed"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if status == "vanished" or not output_mesh:
        o3d.io.write_point_cloud(str(output_path.resolve()), denoised)
    else:
        mesh = reconstruct_mesh(denoised, voxel_size)
        o3d.io.write_triangle_mesh(str(output_path.resolve()), mesh)
        if visualize:
            o3d.visualization.draw_geometries(
                [mesh], window_name=f"Result: {Path(input_path).name}"
            )

    return status


# --- CLI ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Uniform Erosion (UE) denoising for 3D impressions."
    )
    depth = parser.add_mutually_exclusive_group()
    depth.add_argument(
        "--removal_ratio", type=float, default=None,
        help="UE removal ratio (alpha): target neck thickness as a fraction "
             "of the object's maximum side length "
             f"(default: {DEFAULT_REMOVAL_RATIO}, the paper's optimized "
             "value).",
    )
    depth.add_argument(
        "--neck_thickness_mm", type=float, default=None,
        help="Absolute target neck thickness in millimetres (overrides the "
             "ratio-based specification).",
    )
    parser.add_argument(
        "--num_points", type=int, default=DEFAULT_NUM_SAMPLE_POINTS,
        help="Number of points sampled from each input mesh.",
    )
    parser.add_argument(
        "--output_mesh", action="store_true",
        help="Re-mesh the denoised point cloud via alpha-shape "
             "reconstruction instead of writing raw points.",
    )
    parser.add_argument(
        "--input_dir", type=str, default="./data/input",
        help="Root directory containing input meshes/point clouds.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/output_ue",
        help="Root directory where denoised results are written.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show the reconstructed mesh for each file (with --output_mesh).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Default to the paper's optimized removal ratio when neither depth
    # option is given.
    removal_ratio = args.removal_ratio
    neck_thickness_mm = args.neck_thickness_mm
    if removal_ratio is None and neck_thickness_mm is None:
        removal_ratio = DEFAULT_REMOVAL_RATIO

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    files = sorted(
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS
    )

    print("Starting Uniform Erosion (UE) denoising.")
    print(f"  Input root : {input_root}")
    print(f"  Output root: {output_root}")
    if removal_ratio is not None:
        print(f"  removal_ratio={removal_ratio}")
    else:
        print(f"  neck_thickness_mm={neck_thickness_mm}")
    print(f"  Files found: {len(files)}")

    counts = {"ok": 0, "vanished": 0, "failed": 0}
    for input_file in tqdm(files, desc="Processing"):
        relative_path = input_file.relative_to(input_root)
        output_file = (output_root / relative_path).with_suffix(".ply")

        try:
            status = process_file(
                input_file, output_file,
                removal_ratio=removal_ratio,
                neck_thickness_mm=neck_thickness_mm,
                num_points=args.num_points,
                output_mesh=args.output_mesh,
                visualize=args.visualize,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch running
            print(f"\n[Error] Failed: {input_file.name} -> {exc}")
            status = "failed"

        if status == "vanished":
            print(f"\n[Warning] {input_file.name}: erosion removed the "
                  "entire object; original point cloud saved unmodified.")
        counts[status] += 1

    print("-" * 40)
    print("Completed.")
    print(f"  Denoised           : {counts['ok']}")
    print(f"  Vanished (fallback): {counts['vanished']}")
    print(f"  Failed             : {counts['failed']}")


if __name__ == "__main__":
    main()