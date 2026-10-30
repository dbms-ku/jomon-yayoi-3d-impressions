"""Dataset loader for size-aware impression classification (SA-PointNet++).

Adapted from ModelNetDataLoader (ModelNet.py) in the PyTorch
PointNet/PointNet++ implementation by Xu Yan
(yanx27/Pointnet_Pointnet2_pytorch, MIT License). Modifications for this
project:

  * samples are listed in a CSV (columns 'class', 'split', 'path') instead
    of the ModelNet directory convention; point clouds are plain-text XYZ
    files,
  * the axis-aligned bounding-box (AABB) side lengths are computed from
    the raw point cloud BEFORE any sampling or unit-sphere normalization
    (paper Section 4.1) and returned as the Size Feature, and
  * __getitem__ returns (points, size_feature, label, sample_name).

Class indices are assigned alphabetically over the class names found in
the CSV (after stripping trailing digits), so the label order depends on
the naming used in the CSV; keep this in mind when interpreting confusion
matrices. Place this file under data_utils/ of the PointNet/PointNet++
framework.
"""

import os
import pickle
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


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
    duplicated by random choice when undersized.
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


class ImpressionDataset(Dataset):
    """CSV-driven point cloud dataset with AABB size features.

    Each item is (points, size_feature, label, sample_name), where
    size_feature holds the three AABB side lengths of the raw (pre-
    normalization) point cloud in the dataset's physical units (mm).
    """

    def __init__(self, root, args, split="train", process_data=False):
        self.root = root
        self.split = split
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        df = pd.read_csv(args.CSV_FILE)
        df_split = df[df["split"] == self.split]
        shape_paths = df_split["path"].tolist()
        shape_names = df_split["class"].tolist()

        # Strip trailing digits so numbered variants of a class name map to
        # the same label (e.g. "rice2" -> "rice").
        self.datapath = [
            (re.sub(r"\d+$", "", shape_names[i]), shape_paths[i])
            for i in range(len(shape_paths))
        ]
        self.data_names = [
            os.path.splitext(os.path.basename(p[1]))[0] for p in self.datapath
        ]
        class_names = sorted(set(name for name, _ in self.datapath))
        self.classes = {name: i for i, name in enumerate(class_names)}

        for class_name, count in Counter(
            name for name, _ in self.datapath
        ).items():
            print("%s: %d" % (class_name, count))
        print("The size of %s data is %d" % (self.split, len(self.datapath)))

        # Cache name includes "size" to avoid colliding with the cache of
        # the plain (non-size-aware) loader, whose pickle layout differs.
        self.save_path = os.path.join(
            self.root,
            "impression_size_%d_%s_%dpts_fps.dat"
            % (self.num_category, self.split, self.npoints),
        )

        if self.process_data:
            if not os.path.exists(self.save_path):
                print("Processing data %s (only running in the first time)..."
                      % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)
                self.list_of_sizes = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)),
                                  total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = np.array([self.classes[fn[0]]]).astype(np.int32)
                    point_set = load_point_file(fn[1])

                    # AABB size from the raw cloud, before any sampling or
                    # normalization (paper Section 4.1).
                    min_coords = np.min(point_set[:, :3], axis=0)
                    max_coords = np.max(point_set[:, :3], axis=0)
                    size_feat = (max_coords - min_coords).astype(np.float32)

                    point_set = fix_point_count(point_set, self.npoints,
                                                self.uniform)

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls
                    self.list_of_sizes[index] = size_feat

                with open(self.save_path, "wb") as f:
                    pickle.dump([self.list_of_points, self.list_of_labels,
                                 self.list_of_sizes], f)
            else:
                print("Load processed data from %s..." % self.save_path)
                with open(self.save_path, "rb") as f:
                    (self.list_of_points, self.list_of_labels,
                     self.list_of_sizes) = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set = self.list_of_points[index]
            label = self.list_of_labels[index]
            size_feat = self.list_of_sizes[index]
        else:
            fn = self.datapath[index]
            label = np.array([self.classes[fn[0]]]).astype(np.int32)
            point_set = load_point_file(fn[1])

            min_coords = np.min(point_set[:, :3], axis=0)
            max_coords = np.max(point_set[:, :3], axis=0)
            size_feat = (max_coords - min_coords).astype(np.float32)

            point_set = fix_point_count(point_set, self.npoints, self.uniform)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0], size_feat

    def __getitem__(self, index):
        points, label, size_feat = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        if self.split == "train":
            np.random.shuffle(pt_idxs)
        current_points = torch.from_numpy(points[pt_idxs].copy()).float()
        size_tensor = torch.from_numpy(size_feat).float()
        return current_points, size_tensor, label, self.data_names[index]
