"""Dataset loader for the PointNet-based Noise Verification classifier.

JomonNoiseDataset serves the binary noise/positive patches extracted from
the Noise-Pair Dataset (paper Section 4.2.3). pc_normalize and
farthest_point_sample follow the implementations in the PyTorch
PointNet/PointNet++ code by Xu Yan (yanx27/Pointnet_Pointnet2_pytorch,
MIT License).
"""

import os
import numpy as np
import warnings
import pickle
import open3d as o3d  # used for reading PLY files
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def adjust_point_count(points, npoint):
    """
    Fix the number of points to npoint.
    If there are fewer points than npoint, points are randomly resampled
    with replacement; if there are more, they are downsampled with FPS.
    """
    N, D = points.shape
    if N >= npoint:
        # Too many points: downsample with farthest point sampling
        return farthest_point_sample(points, npoint)
    else:
        # Too few points: randomly duplicate points to reach npoint
        choice = np.random.choice(N, npoint, replace=True)
        return points[choice, :]


def farthest_point_sample(point, npoint):
    N, D = point.shape
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


class JomonNoiseDataset(Dataset):
    def __init__(self, root, npoints=256, split='train', args=None, **kwargs):
        """
        root: parent directory containing 'positive_dataset_all' and
              'noise_dataset_all'
        npoints: number of points per patch (used when args is None)
        split: 'train' or 'test'
        args: training script arguments; if given, args.num_point is used
              as the number of points per patch
        Additional keyword arguments (e.g. process_data) are accepted for
        compatibility with the training script and are ignored.
        """
        self.root = root
        self.npoints = args.num_point if args is not None else npoints

        # Class definitions
        self.classes = {'noise': 0, 'positive': 1}
        self.datapath = []

        # Build the file list
        for cat in self.classes.keys():
            dir_path = os.path.join(self.root, f"{cat}_dataset_all")
            if not os.path.exists(dir_path):
                print(f"Warning: {dir_path} not found.")
                continue

            files = os.listdir(dir_path)
            # Simple train/test split (80:20) on the sorted file list
            files.sort()
            num_files = len(files)
            if split == 'train':
                selected_files = files[:int(num_files * 0.8)]
            else:
                selected_files = files[int(num_files * 0.8):]

            for f in selected_files:
                if f.endswith('.ply'):
                    self.datapath.append((cat, os.path.join(dir_path, f)))

        print(f'The size of {split} data is {len(self.datapath)}')

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        cat, filepath = self.datapath[index]
        label = np.array([self.classes[cat]]).astype(np.int32)

        # Load the PLY file with Open3D
        pcd = o3d.io.read_point_cloud(filepath)
        point_set = np.asarray(pcd.points).astype(np.float32)

        # Fix the number of points
        point_set = adjust_point_count(point_set, self.npoints)

        # Normalize
        point_set = pc_normalize(point_set)

        # Sample name (file name without extension)
        data_name = os.path.splitext(os.path.basename(filepath))[0]

        return point_set, label[0], data_name

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    # Example usage
    # dataset = JomonNoiseDataset('./data/', npoints=256)
    # print(dataset[0][0].shape)
    pass