from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        name: str = "efficientnet_b4",
        out_indices: List[int] = (1, 2, 3, 4),
        pretrained: bool = True,
        target_size: int = 14,
    ):
        super().__init__()
        self.target_size = target_size
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.feature_channels = self.backbone.feature_info.channels()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        resized = [
            F.interpolate(f, size=self.target_size, mode="bilinear", align_corners=False)
            for f in feats
        ]
        return torch.cat(resized, dim=1)
