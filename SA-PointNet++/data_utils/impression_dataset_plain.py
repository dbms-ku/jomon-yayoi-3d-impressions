"""Dataset loader for baseline (non-size-aware) impression classification.

Adapted from ModelNetDataLoader (ModelNet.py) in the PyTorch
PointNet/PointNet++ implementation by Xu Yan
(yanx27/Pointnet_Pointnet2_pytorch, MIT License). Modifications for this
project:

  * samples are listed in a CSV (columns 'class', 'split', 'path') instead
    of the ModelNet directory convention; point clouds are plain-text XYZ
    files, and
  * __getitem__ returns (points, label, sample_name).

This is the loader for the plain PointNet++ baseline; it does NOT compute
the AABB Size Feature. Use ImpressionDataset (impression_dataset.py) for
the size-aware models. The two loaders use distinct cache filenames
because their pickle layouts are incompatible.

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


class ImpressionDatasetPlain(Dataset):
    """CSV-driven point cloud dataset for the size-agnostic baseline.

    Each item is (points, label, sample_name); points are unit-sphere
    normalized as in the standard PointNet++ pipeline.
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

        # Distinct cache name from the size-aware loader (incompatible
        # pickle layouts).
        self.save_path = os.path.join(
            self.root,
            "impression_%d_%s_%dpts_fps.dat"
            % (self.num_category, self.split, self.npoints),
        )

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

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        if self.split == "train":
            np.random.shuffle(pt_idxs)
        current_points = torch.from_numpy(points[pt_idxs].copy()).float()
        return current_points, label, self.data_names[index]
