"""SA-PointNeXt: size-aware classification wrapper for the openpoints framework.

Adapted from BaseCls (openpoints/models/classification/cls_base.py) in the
official PointNeXt implementation (guochengqian/PointNeXt, MIT License).
This wrapper adds the same late-fusion size-awareness as SA-PointNet++
(paper Sections 4.1 and 6.6, Table 8): the AABB side lengths are
concatenated with the encoder's global feature after pooling, and the
classification head's input width is enlarged accordingly.

The dataloader must provide the size feature in the input dict under the
key 'aabb_size' with shape (B, size_feature_dim), computed from the raw
point cloud before normalization.

Installation into the PointNeXt framework:
  1. place this file at openpoints/models/classification/cls_base_size.py
  2. add `from .cls_base_size import BaseClsSize` to
     openpoints/models/classification/__init__.py
  3. reference the model in the experiment config as NAME: BaseClsSize
"""

import torch
import torch.nn as nn

from ..build import MODELS, build_model_from_cfg
from ...loss import build_criterion_from_cfg


@MODELS.register_module()
class BaseClsSize(nn.Module):
    """PointNeXt classifier with AABB size features fused after pooling."""

    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 criterion_args=None,
                 size_feature_dim=3,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        self.size_feature_dim = size_feature_dim

        if cls_args is not None:
            # Widen the head input by the size feature's dimensionality.
            in_channels = self.encoder.out_channels if hasattr(
                self.encoder, 'out_channels') else cls_args.get(
                'in_channels', None)
            if in_channels is None:
                raise ValueError(
                    "BaseClsSize could not determine the encoder output "
                    "width; set cls_args.in_channels explicitly.")
            cls_args.in_channels = in_channels + size_feature_dim
            self.prediction = build_model_from_cfg(cls_args)
        else:
            self.prediction = nn.Identity()

        self.criterion = build_criterion_from_cfg(
            criterion_args) if criterion_args is not None else None

    def forward(self, data):
        # Global feature from the PointNeXt encoder.
        global_feat = self.encoder.forward_cls_feat(data)

        # Late fusion with the AABB size feature (paper Section 4.1).
        size_feat = data.get('aabb_size', None)
        if size_feat is None:
            raise KeyError(
                "BaseClsSize requires 'aabb_size' in the input dict "
                "(shape (B, size_feature_dim)); the dataloader must "
                "compute it from the raw point cloud before "
                "normalization.")
        global_feat = torch.cat([global_feat, size_feat], dim=1)

        return self.prediction(global_feat)

    def get_loss(self, pred, gt, inputs=None):
        return self.criterion(pred, gt.long())

    def get_logits_loss(self, data, gt):
        logits = self.forward(data)
        return logits, self.criterion(logits, gt.long())