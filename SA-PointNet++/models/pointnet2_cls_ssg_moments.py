"""Moments-PointNet++ classification model (statistical-moment ablation).

Adapted from pointnet2_cls_ssg.py in the PyTorch PointNet/PointNet++
implementation by Xu Yan (yanx27/Pointnet_Pointnet2_pytorch, MIT License).
This is the moment-based ablation of SA-PointNet++ used for the size
feature comparison (paper Section 6.3, Table 5): instead of the 3-dim
AABB side lengths, a 12-dim statistical-moment vector (mean, variance,
skewness, and kurtosis along each axis) is concatenated with the 1024-dim
global shape descriptor after max-pooling.

Apart from the auxiliary feature's name and dimensionality, the
architecture is identical to pointnet2_cls_ssg_size.py, so any
performance difference between the two models is attributable to the
choice of auxiliary feature alone.

The forward pass returns (log-probabilities, l3_points). Place this file
under models/ of the PointNet/PointNet++ framework as
pointnet2_cls_ssg_moments.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_model_moments(nn.Module):
    def __init__(self, num_class, normal_channel=True, moment_feature_dim=12):
        super(get_model_moments, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                          in_channel=in_channel,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128 + 3,
                                          mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None,
                                          nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        # Moment-based modification: the classification head receives the
        # global shape descriptor concatenated with the moment feature.
        self.fc1 = nn.Linear(1024 + moment_feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, moment_features):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # 1024-dim global shape descriptor (after max-pooling) concatenated
        # with the 12-dim statistical-moment feature (paper Section 6.3).
        shape_feat = l3_points.view(B, 1024)
        combined_feat = torch.cat([shape_feat, moment_features], dim=1)

        x = self.drop1(F.relu(self.bn1(self.fc1(combined_feat))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss