import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .flash_encoder import FLASHEncoder

class HDCMLPEncoder(nn.Module):
    """
    CNN backbone + FLASH HDC encoder + MLP head.
    Outputs L2-normalized visual embeddings aligned with text embeddings.
    """
    def __init__(
        self,
        cnn_name: str = "resnet18",
        hd_dim: int = 4096,
        text_feat_dim: int = 768,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        # Backbone CNN
        base = getattr(models, cnn_name)(pretrained=pretrained_backbone)
        # Remove classifier, keep global pooled feature
        if cnn_name.startswith("resnet"):
            in_dim = base.fc.in_features
            base.fc = nn.Identity()
        else:
            raise NotImplementedError("Only ResNet-like backbones handled in this template.")
        self.backbone = base

        # FLASH HDC encoder
        self.flash = FLASHEncoder(in_dim=in_dim, hd_dim=hd_dim)

        # MLP head: HDC hypervector -> visual embedding
        self.mlp = nn.Sequential(
            nn.Linear(hd_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, text_feat_dim),
        )

    def encode_images(self, images):
        """
        images: [B, 3, H, W]
        returns: visual embeddings [B, text_feat_dim]
        """
        feats = self.backbone(images)       # [B, in_dim]
        h = self.flash(feats)               # [B, hd_dim]
        out = self.mlp(h)                   # [B, text_feat_dim]
        out = F.normalize(out, p=2, dim=-1)
        return out

    def forward(self, images):
        return self.encode_images(images)
