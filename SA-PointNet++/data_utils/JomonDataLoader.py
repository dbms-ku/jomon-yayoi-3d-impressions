"""Dataset loader for the Jomon-Yayoi evaluation set (non-size-aware).

Adapted from ModelNetDataLoader (ModelNet.py) in the PyTorch
PointNet/PointNet++ implementation by Xu Yan
(yanx27/Pointnet_Pointnet2_pytorch, MIT License). Modifications for this
project:

  * samples are listed in per-split text files (one point-cloud file path
    per line) with a separate class-name catalog file, and the class of
    each sample is derived from its parent directory name, and
  * __getitem__ returns (points, label, sample_name).

This is the loader for the plain PointNet++ baseline on Jomon-Yayoi data;
it does NOT compute the AABB Size Feature. Use JomonYayoiDataset
(jomon_yayoi_dataset.py) for the size-aware models. The two loaders use
distinct cache filenames because their pickle layouts are incompatible.

Label consistency: class indices follow the LINE ORDER of the catalog
file, which must match the alphabetical order used by the Experimental
training loaders (see jomon_yayoi_dataset.py for details).

Expected files under `root` (names configurable via optional attributes
on `args`: shape_names_file, split_file_template):
  * jomon_shape_names.txt          one class name per line (label order)
  * jomon_{train,val,test}.txt     one point-cloud file path per line

Place this file under data_utils/ of the PointNet/PointNet++ framework.
"""

import os
import pickle
import warnings

import numpy as np
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


def load_point_file(path):
    """Load a plain-text point cloud, auto-detecting the delimiter.

    Accepts whitespace- or comma-separated XYZ(+extra) columns.
    """
    with open(path, "r") as fh:
        first_line = fh.readline()
    delimiter = "," if "," in first_line else None  # None = any whitespace
    return np.loadtxt(path, delimiter=delimiter).astype(np.float32)


def fix_point_count(point_set, npoints, uniform):
    """Bring a point set to exactly npoints points.

    With uniform=True, Farthest Point Sampling is used. Otherwise the
    first npoints points are taken when oversized, and points are
    duplicated by random choice when undersized (oversampling with
    replacement, paper Section 5.1).
    """
    if uniform:
        return farthest_point_sample(point_set, npoints)

    num_points = point_set.shape[0]
    if num_points > npoints:
        return point_set[0:npoints, :]
    if num_points < npoints:
        indices = np.random.choice(num_points, npoints, replace=True)
        return point_set[indices, :]
    return point_set


class JomonYayoiDatasetPlain(Dataset):
    """Split-file-driven Jomon-Yayoi dataset for the size-agnostic baseline.

    Each item is (points, label, sample_name); points are unit-sphere
    normalized as in the standard PointNet++ pipeline.
    """

    def __init__(self, root, args, split="train", process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

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

        self.data_names = [
            os.path.splitext(os.path.basename(p[1]))[0] for p in self.datapath
        ]

        # Distinct cache name from the size-aware loader (incompatible
        # pickle layouts).
        if self.uniform:
            self.save_path = os.path.join(
                root, "jomon_yayoi_%d_%s_%dpts_fps.dat"
                % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(
                root, "jomon_yayoi_%d_%s_%dpts.dat"
                % (self.num_category, split, self.npoints))
        print("process data is %s" % process_data)

        if self.process_data:
            if not os.path.exists(self.save_path):
                print("Processing data %s (only running in the first time)..."
                      % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)),
                                  total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = np.array([self.classes[fn[0]]]).astype(np.int32)
                    point_set = load_point_file(fn[1])
                    point_set = fix_point_count(point_set, self.npoints,
                                                self.uniform)

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, "wb") as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print("Load processed data from %s..." % self.save_path)
                with open(self.save_path, "rb") as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set = self.list_of_points[index]
            label = self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            label = np.array([self.classes[fn[0]]]).astype(np.int32)
            point_set = load_point_file(fn[1])
            point_set = fix_point_count(point_set, self.npoints, self.uniform)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0], self.data_names[index]

    def __getitem__(self, index):
        return self._get_item(index)


# Backward-compatible alias for scripts written against the original name.
JomonDataLoader = JomonYayoiDatasetPlain