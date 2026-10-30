"""Size-Aware PointNet++ (SA-PointNet++) classification model.

Adapted from pointnet2_cls_ssg.py in the PyTorch PointNet/PointNet++
implementation by Xu Yan (yanx27/Pointnet_Pointnet2_pytorch, MIT License).
This file implements the size-aware classifier proposed in the paper
(Section 4.1, Figure 6). Modifications from the upstream model:

  * the factory class is named get_model_size and takes an additional
    size_feature_dim argument (3 for the AABB side lengths),
  * forward takes (xyz, size_features); after the Set Abstraction layers
    and max-pooling produce the 1024-dim global shape descriptor, the
    size feature is concatenated to form a (1024 + size_feature_dim)-dim
    integrated vector, and
  * the first fully connected layer accordingly accepts
    1024 + size_feature_dim inputs; the classification head
    (512 -> 256 -> num_class) is unchanged.

The forward pass returns (log-probabilities, l3_points); l3_points is the
pre-concatenation global feature, which the evaluation scripts export for
the t-SNE visualization (paper Figure 11). Place this file under models/
of the PointNet/PointNet++ framework as pointnet2_cls_ssg_size.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_model_size(nn.Module):
    def __init__(self, num_class, normal_channel=True, size_feature_dim=3):
        super(get_model_size, self).__init__()
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
        # Size-aware modification: the classification head receives the
        # global shape descriptor concatenated with the size feature.
        self.fc1 = nn.Linear(1024 + size_feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, size_features):
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
        # with the AABB size feature -> (1024 + size_feature_dim)-dim
        # integrated feature vector (paper Section 4.1, Figure 6).
        shape_feat = l3_points.view(B, 1024)
        combined_feat = torch.cat([shape_feat, size_features], dim=1)

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