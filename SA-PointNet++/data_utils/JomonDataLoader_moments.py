"""Dataset loader for the Jomon-Yayoi set with statistical-moment features.

Adapted from ModelNetDataLoader (ModelNet.py) in the PyTorch
PointNet/PointNet++ implementation by Xu Yan
(yanx27/Pointnet_Pointnet2_pytorch, MIT License). This is the
moment-based counterpart of jomon_yayoi_dataset.py (paper Section 6.3,
Table 5): instead of the 3-dim AABB side lengths, a 12-dim statistical-
moment vector (mean, variance, skewness, kurtosis along each axis) is
computed from the raw point cloud BEFORE any sampling or unit-sphere
normalization and returned as the auxiliary feature.

__getitem__ returns (points, moment_feature, label, sample_name), so the
loader is a drop-in replacement for JomonYayoiDataset when evaluating
get_model_moments (pointnet2_cls_ssg_moments.py).

Label order: class indices follow the LINE ORDER of the catalog file,
which must match the alphabetical class order produced by the
Experimental training loader (see jomon_yayoi_dataset.py for details;
this correspondence has been verified for the released catalog).

Expected files under `root` (names configurable via optional attributes
on `args`: shape_names_file, split_file_template):
  * jomon_shape_names.txt          one class name per line (label order)
  * jomon_{train,val,test}.txt     one point-cloud file path per line

Place this file under data_utils/ of the PointNet/PointNet++ framework.
"""

import os
import pickle
import warnings
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Default catalog / split file names (override via args attributes).
DEFAULT_SHAPE_NAMES_FILE = "jomon_shape_names.txt"
DEFAULT_SPLIT_FILE_TEMPLATE = "jomon_{split}.txt"


def pc_normalize(pc):
    """Center a point set and scale it into the unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / m


def farthest_point_sample(point, npoint, index=None, fn=None):
    """Subsample a point set to npoint points via Farthest Point Sampling.

    Warns when the input has fewer points than requested (indices are then
    duplicated), which can happen for heavily denoised specimens.
    """
    N, _ = point.shape
    if N < npoint:
        print(f"Warning: npoint({npoint}) > N({N}) "
              f"for index {index}, file {fn}")

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


def load_point_file(path):
    """Load a plain-text point cloud, auto-detecting the delimiter.

    Accepts whitespace- or comma-separated XYZ(+extra) columns.
    """
    with open(path, "r") as fh:
        first_line = fh.readline()
    delimiter = "," if "," in first_line else None  # None = any whitespace
    return np.loadtxt(path, delimiter=delimiter).astype(np.float32)


def fix_point_count(point_set, npoints, uniform, index=None, fn=None):
    """Bring a point set to exactly npoints points.

    With uniform=True, Farthest Point Sampling is used. Otherwise the
    first npoints points are taken when oversized, and points are
    duplicated by random choice when undersized.
    """
    if uniform:
        return farthest_point_sample(point_set, npoints, index, fn)

    num_points = point_set.shape[0]
    if num_points > npoints:
        return point_set[0:npoints, :]
    if num_points < npoints:
        indices = np.random.choice(num_points, npoints, replace=True)
        return point_set[indices, :]
    return point_set


def compute_statistical_moments(point_set_np):
    """Compute a 12-dim statistical-moment feature from raw coordinates.

    For each axis (x, y, z), the mean, variance, skewness, and kurtosis
    are computed, yielding 3 axes x 4 moments = 12 dimensions. Call this
    on the raw (pre-normalization) point cloud so that absolute-scale
    information is retained.
    """
    xyz = point_set_np[:, :3].astype(np.float64)

    moment_feat = []
    for axis in range(3):
        vals = xyz[:, axis]
        m1 = np.mean(vals)                 # mean
        m2 = np.var(vals)                  # variance
        std = np.std(vals)
        if std < 1e-8:
            m3 = 0.0                       # skewness (degenerate: collinear)
            m4 = 0.0                       # kurtosis
        else:
            m3 = np.mean(((vals - m1) / std) ** 3)
            m4 = np.mean(((vals - m1) / std) ** 4)
        moment_feat.extend([m1, m2, m3, m4])

    return np.array(moment_feat, dtype=np.float32)


class JomonYayoiDatasetMoments(Dataset):
    """Split-file-driven Jomon-Yayoi dataset with 12-dim moment features.

    Each item is (points, moment_feature, label, sample_name), where
    moment_feature holds the 12 statistical moments of the raw (pre-
    normalization) point cloud.
    """

    def __init__(self, root, args, split="train", process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.subset = split

        assert split in ("train", "test", "val")

        shape_names_file = getattr(
            args, "shape_names_file", DEFAULT_SHAPE_NAMES_FILE
        )
        split_template = getattr(
            args, "split_file_template", DEFAULT_SPLIT_FILE_TEMPLATE
        )

        # Class order follows the catalog file's line order (see the module
        # docstring for the label-consistency requirement).
        self.catfile = os.path.join(self.root, shape_names_file)
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        split_path = os.path.join(
            self.root, split_template.format(split=split)
        )
        split_lines = [line.rstrip() for line in open(split_path)]

        # Each line is a full path to a point-cloud file; the class name is
        # the parent directory name. macOS metadata entries are skipped.
        self.datapath = []
        for line in split_lines:
            if ".DS" in line:
                continue
            class_name = os.path.basename(os.path.dirname(line))
            self.datapath.append((class_name, line))

        print("The size of %s data is %d" % (split, len(self.datapath)))

        # Sample names keep the class subfolder for uniqueness across classes.
        self.data_names = [
            os.path.join(
                os.path.basename(os.path.dirname(p[1])),
                os.path.splitext(os.path.basename(p[1]))[0],
            )
            for p in self.datapath
        ]

        for class_name, count in Counter(
            name for name, _ in self.datapath
        ).items():
            print("%s: %d" % (class_name, count))

        # Distinct cache name from the AABB-size loader (both pickle three
        # lists, but the auxiliary feature differs in dimensionality).
        if self.uniform:
            self.save_path = os.path.join(
                root, "jomon_yayoi_moments_%d_%s_%dpts_fps.dat"
                % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(
                root, "jomon_yayoi_moments_%d_%s_%dpts.dat"
                % (self.num_category, split, self.npoints))
        print("process data is %s" % process_data)

        if self.process_data:
            if not os.path.exists(self.save_path):
                print("Processing data %s (only running in the first time)..."
                      % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)
                self.list_of_moments = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)),
                                  total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = np.array([self.classes[fn[0]]]).astype(np.int32)
                    point_set = load_point_file(fn[1])

                    # Moments from the raw cloud, before any sampling or
                    # normalization (paper Section 6.3).
                    moment_feat = compute_statistical_moments(point_set)

                    point_set = fix_point_count(
                        point_set, self.npoints, self.uniform, index, fn
                    )

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls
                    self.list_of_moments[index] = moment_feat

                with open(self.save_path, "wb") as f:
                    pickle.dump([self.list_of_points, self.list_of_labels,
                                 self.list_of_moments], f)
            else:
                print("Load processed data from %s..." % self.save_path)
                with open(self.save_path, "rb") as f:
                    (self.list_of_points, self.list_of_labels,
                     self.list_of_moments) = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set = self.list_of_points[index]
            label = self.list_of_labels[index]
            moment_feat = self.list_of_moments[index]
        else:
            fn = self.datapath[index]
            label = np.array([self.classes[fn[0]]]).astype(np.int32)
            point_set = load_point_file(fn[1])

            moment_feat = compute_statistical_moments(point_set)

            point_set = fix_point_count(
                point_set, self.npoints, self.uniform, index, fn
            )

        # Normalization happens AFTER the moment feature is computed.
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0], moment_feat, self.data_names[index]

    def __getitem__(self, index):
        points, label, moment_feat, name = self._get_item(index)

        pt_idxs = np.arange(0, points.shape[0])
        if self.subset == "train":
            np.random.shuffle(pt_idxs)
        current_points = torch.from_numpy(points[pt_idxs].copy()).float()
        moment_tensor = torch.from_numpy(moment_feat).float()

        return current_points, moment_tensor, label, name


# Backward-compatible alias for scripts written against the original name.
JomonDataLoader_moments = JomonYayoiDatasetMoments