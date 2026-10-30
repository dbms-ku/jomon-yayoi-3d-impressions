# openpoints/dataset/impression/impression_dataset.py

import os
import numpy as np
import pandas as pd
import re
import logging
from collections import Counter
from torch.utils.data import Dataset
from ..build import DATASETS

JAPANESE_CLASS_MAP = {
    'アワ':           'foxtail_millet',
    'イネ':           'rice',
    'エゴマ':         'perilla',
    'カラスザンショウ': 'ailanthoides',
    'コクゾウムシ':   'maize_weevil',
}

@DATASETS.register_module()
class ImpressionDataset(Dataset):
    """
    Dataset for archaeological impression classification.
    Loads point clouds from .txt files specified in a CSV file.
    """
    def __init__(self,
                 data_dir,
                 csv_file,
                 split='train',
                 num_points=8192,
                 transform=None,
                 **kwargs):
        self.num_points = num_points
        self.transform  = transform
        self.split      = split

        # CSV読み込み
        df = pd.read_csv(csv_file)
        # split列の正規化（valid/valどちらも対応）
        df['split'] = df['split'].str.strip().str.lower()
        df['split'] = df['split'].replace({'val': 'valid'})

        split_key = 'valid' if split.lower() in ('val', 'valid') else split.lower()
        df_split  = df[df['split'] == split_key]

        # クラス名の正規化（末尾の数字を除去）
        df_split = df_split.copy()
        df_split['class_en'] = df_split['class'].apply(
            lambda x: JAPANESE_CLASS_MAP.get(
                re.sub(r'\d+$', '', str(x).strip()), None
            )
        )
        df_split = df_split[df_split['class_en'].notna()]

        # クラスインデックスの作成
        class_names      = sorted(df_split['class_en'].unique())
        self.classes     = {name: i for i, name in enumerate(class_names)}
        self.class_names = class_names

        # パスとラベルのリスト作成
        self.data_paths = []
        self.labels     = []

        csv_dir = os.path.dirname(os.path.abspath(csv_file))
        for _, row in df_split.iterrows():
            path = str(row['path']).strip()
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(csv_dir, path))
            if not os.path.exists(path):
                logging.warning(f'File not found: {path}')
                continue
            self.data_paths.append(path)
            self.labels.append(self.classes[row['class_en']])

        # クラス分布のログ出力
        counter = Counter(self.labels)
        for name, idx in self.classes.items():
            logging.info(f'  {name}: {counter.get(idx, 0)} samples')
        logging.info(f'==> loaded {split} data: {len(self.data_paths)} samples')

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # 点群の読み込み
        points = np.loadtxt(self.data_paths[idx], delimiter=' ').astype(np.float32)

        # AABBサイズを正規化前に計算
        min_coords = points[:, :3].min(axis=0)
        max_coords = points[:, :3].max(axis=0)
        aabb_size  = (max_coords - min_coords).astype(np.float32)

        # 点数の調整
        n = points.shape[0]
        # if n >= self.num_points:
        #     if self.split == 'train':
        #         idx_choice = np.random.choice(n, self.num_points, replace=False)
        #     else:
        #         np.random.seed(idx)
        #         idx_choice = np.random.choice(n, self.num_points, replace=False)
        # else:
        #     idx_choice = np.random.choice(n, self.num_points, replace=True)
        if n >= self.num_points:
            if self.split == 'train':
                idx_choice = np.random.choice(n, self.num_points, replace=False)
            else:
                # シード固定を外す → 毎回異なるサンプリング
                idx_choice = np.random.choice(n, self.num_points, replace=False)
        else:
            idx_choice = np.random.choice(n, self.num_points, replace=True)
        points = points[idx_choice, :3]

        # 正規化
        centroid = points.mean(axis=0)
        points   = points - centroid
        m        = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if m > 1e-8:
            points = points / m

        # transformにはpos・yのみ渡す（aabb_sizeは除外）
        data = {
            'pos': points,
            'y':   np.int32(self.labels[idx]),  # スカラー値
        }

        if self.transform is not None:
            data = self.transform(data)

        import torch
        # x特徴量の設定
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']

        # aabb_sizeはtransform後に追加
        data['aabb_size'] = torch.from_numpy(aabb_size).float()

        return data

    @property
    def num_classes(self):
        return len(self.classes)