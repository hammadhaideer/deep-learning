import torch
import torch.nn as nn
from einops import rearrange

from .backbone import EfficientNetFeatureExtractor
from .transformer import UniADTransformer


class UniAD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_size = cfg["data"]["feature_size"]
        self.image_size = cfg["data"]["image_size"]

        self.backbone = EfficientNetFeatureExtractor(
            name=cfg["backbone"]["name"],
            out_indices=cfg["backbone"]["out_indices"],
            pretrained=cfg["backbone"]["pretrained"],
            target_size=self.feature_size,
        )
        feat_dim = sum(self.backbone.feature_channels)
        assert feat_dim == cfg["backbone"]["feature_dim"], (
            f"Backbone feature dim {feat_dim} != config {cfg['backbone']['feature_dim']}"
        )

        self.transformer = UniADTransformer(
            feature_dim=feat_dim,
            hidden_dim=cfg["model"]["hidden_dim"],
            num_heads=cfg["model"]["num_heads"],
            num_encoder_layers=cfg["model"]["num_encoder_layers"],
            num_decoder_layers=cfg["model"]["num_decoder_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"],
            feature_size=self.feature_size,
            neighbor_mask_size=cfg["model"]["neighbor_mask_size"],
            jitter_scale=cfg["model"]["jitter_scale"],
            jitter_prob=cfg["model"]["jitter_prob"],
            layer_wise_query=cfg["model"]["layer_wise_query"],
        )

    def extract_tokens(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image)
        return rearrange(feats, "b c h w -> b (h w) c")

    def forward(self, image: torch.Tensor):
        tokens = self.extract_tokens(image)
        recon = self.transformer(tokens)
        return tokens, recon

    @torch.no_grad()
    def anomaly_map(self, image: torch.Tensor) -> torch.Tensor:
        tokens, recon = self.forward(image)
        diff = (tokens - recon).pow(2).mean(dim=-1)
        diff = rearrange(diff, "b (h w) -> b 1 h w", h=self.feature_size, w=self.feature_size)
        diff = torch.nn.functional.interpolate(
            diff, size=self.image_size, mode="bilinear", align_corners=False
        )
        return diff.squeeze(1)
