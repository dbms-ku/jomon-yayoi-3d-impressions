"""Apply optimized SOR/ROR denoising to the Jomon-Yayoi dataset.

This script applies Statistical Outlier Removal (SOR) and Radius Outlier
Removal (ROR) to every specimen of the Jomon-Yayoi dataset using the
parameters found by optimize_sor_ror.py (paper Section 6.5). The denoised
point clouds are written per class as plain-text XYZ files so they can be
fed to the downstream classifier.

The class folders are read directly from the released dataset layout; pass
the class subfolder names via --class_dirs if they differ from the English
class codes used below.

Example:
    python denoise_sor_ror.py \
        --jy_root path/to/jomon_yayoi \
        --output_root path/to/output \
        --sor_nb_neighbors 50 --sor_std_ratio 1.3 \
        --ror_nb_points 21 --ror_radius_ratio 0.025
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

# --- Constants ---------------------------------------------------------------

# Number of points sampled from each input mesh (Poisson-disk sampling).
DEFAULT_SAMPLE_POINTS = 50000

# The five class subfolder names in the released dataset layout.
DEFAULT_CLASS_DIRS = [
    "foxtail_millet", "rice", "perilla", "ailanthoides", "maize_weevil",
]


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


# --- Denoising ---------------------------------------------------------------

def load_stl_to_pcd(stl_path, n_points=DEFAULT_SAMPLE_POINTS):
    """Sample a point cloud from an STL mesh."""
    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    return mesh.sample_points_poisson_disk(number_of_points=n_points)


def object_max_size(pcd):
    """Longest axis-aligned bounding-box side length of a point cloud."""
    bbox = pcd.get_axis_aligned_bounding_box()
    return float(np.max(bbox.get_max_bound() - bbox.get_min_bound()))


def process_file(stl_path, out_sor_dir, out_ror_dir,
                 sor_nb_neighbors, sor_std_ratio,
                 ror_nb_points, ror_radius_ratio,
                 n_points=DEFAULT_SAMPLE_POINTS):
    """Apply SOR and ROR to one file, saving each result as an XYZ text file."""
    pcd = load_stl_to_pcd(stl_path, n_points=n_points)
    original_count = len(pcd.points)
    max_size = object_max_size(pcd)
    stem = Path(stl_path).stem

    pcd_sor, _ = pcd.remove_statistical_outlier(
        nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio
    )
    np.savetxt(str(out_sor_dir / f"{stem}.txt"),
               np.asarray(pcd_sor.points), fmt="%.6f", delimiter=" ")

    radius = max_size * ror_radius_ratio
    pcd_ror, _ = pcd.remove_radius_outlier(
        nb_points=ror_nb_points, radius=radius
    )
    np.savetxt(str(out_ror_dir / f"{stem}.txt"),
               np.asarray(pcd_ror.points), fmt="%.6f", delimiter=" ")

    return {
        "original": original_count,
        "sor": len(pcd_sor.points),
        "ror": len(pcd_ror.points),
    }


# --- CLI ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply optimized SOR/ROR denoising to Jomon-Yayoi data."
    )
    parser.add_argument("--jy_root", type=str, required=True,
                        help="Root directory of the Jomon-Yayoi dataset.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output root; SOR/ and ROR/ subtrees are created.")
    parser.add_argument("--class_dirs", type=str, nargs="+",
                        default=DEFAULT_CLASS_DIRS,
                        help="Class subfolder names to process.")
    parser.add_argument("--n_points", type=int, default=DEFAULT_SAMPLE_POINTS,
                        help="Points sampled per mesh.")
    parser.add_argument("--sor_nb_neighbors", type=int, required=True,
                        help="SOR: number of neighbors.")
    parser.add_argument("--sor_std_ratio", type=float, required=True,
                        help="SOR: standard-deviation multiplier.")
    parser.add_argument("--ror_nb_points", type=int, required=True,
                        help="ROR: minimum neighbors within the radius.")
    parser.add_argument("--ror_radius_ratio", type=float, required=True,
                        help="ROR: radius as a fraction of the object's max "
                             "side length.")
    parser.add_argument("--log_dir", type=str, default=".",
                        help="Directory where the log file is written.")
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_path = Path(args.log_dir) / f"apply_sor_ror_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as log_fh:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.__stdout__, log_fh)
        try:
            print(f"[INFO] Log file    : {log_path}")
            print(f"[INFO] Timestamp   : "
                  f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"[INFO] jy_root     : {args.jy_root}")
            print(f"[INFO] output_root : {args.output_root}")
            print(f"[INFO] SOR nb_neighbors={args.sor_nb_neighbors}, "
                  f"std_ratio={args.sor_std_ratio}")
            print(f"[INFO] ROR nb_points={args.ror_nb_points}, "
                  f"radius_ratio={args.ror_radius_ratio}")

            jy_root = Path(args.jy_root)
            output_root = Path(args.output_root)
            all_results = []
            summary = {}

            for class_name in args.class_dirs:
                class_dir = jy_root / class_name
                if not class_dir.is_dir():
                    print(f"  [WARN] Missing class folder (skipped): "
                          f"{class_dir}")
                    continue

                stl_files = sorted(
                    list(class_dir.glob("*.stl"))
                    + list(class_dir.glob("*.STL"))
                )
                if not stl_files:
                    print(f"  [WARN] No STL files (skipped): {class_dir}")
                    continue

                out_sor = output_root / "SOR" / class_name
                out_ror = output_root / "ROR" / class_name
                out_sor.mkdir(parents=True, exist_ok=True)
                out_ror.mkdir(parents=True, exist_ok=True)

                print(f"\n[{class_name}]  {len(stl_files)} files")
                class_results = []
                for stl_path in tqdm(stl_files):
                    result = process_file(
                        stl_path, out_sor, out_ror,
                        sor_nb_neighbors=args.sor_nb_neighbors,
                        sor_std_ratio=args.sor_std_ratio,
                        ror_nb_points=args.ror_nb_points,
                        ror_radius_ratio=args.ror_radius_ratio,
                        n_points=args.n_points,
                    )
                    class_results.append(result)
                    all_results.append(result)

                summary[class_name] = {
                    "orig": np.mean([r["original"] for r in class_results]),
                    "sor": np.mean([r["sor"] for r in class_results]),
                    "ror": np.mean([r["ror"] for r in class_results]),
                }
                print(f"  Mean points  original: {summary[class_name]['orig']:.0f}"
                      f"  SOR: {summary[class_name]['sor']:.0f}"
                      f"  ROR: {summary[class_name]['ror']:.0f}")

            print("\n" + "=" * 60)
            print("Summary (mean points per class)")
            print("=" * 60)
            print(f"{'Class':<20} {'Original':>10} {'SOR':>10} {'ROR':>10}")
            print("-" * 60)
            for class_name, s in summary.items():
                print(f"{class_name:<20} {s['orig']:>10.0f} "
                      f"{s['sor']:>10.0f} {s['ror']:>10.0f}")
            if all_results:
                print("-" * 60)
                print(f"{'Overall mean':<20} "
                      f"{np.mean([r['original'] for r in all_results]):>10.0f} "
                      f"{np.mean([r['sor'] for r in all_results]):>10.0f} "
                      f"{np.mean([r['ror'] for r in all_results]):>10.0f}")

            print(f"\nOutput: {output_root}")
            print(f"  SOR -> {output_root / 'SOR'}")
            print(f"  ROR -> {output_root / 'ROR'}")
        finally:
            sys.stdout = original_stdout

    print(f"[INFO] Log saved to: {log_path}")


if __name__ == "__main__":
    main()