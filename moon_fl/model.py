import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity as sim
import timm


class ViTEncoder(nn.Module):
    """
    Lightweight ViT encoder using timm.
    deit_tiny_patch16_224 gives strong features at low compute cost.
    For heavier compute: vit_small_patch16_224 or vit_base_patch16_224
    """

    def __init__(self, model_name='deit_tiny_patch16_224', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # remove classification head
            global_pool='avg'    # average pool CLS token
        )
        self.out_dim = self.backbone.num_features  # 192 for deit_tiny

    def forward(self, x):
        # timm expects 224x224 — ensure resize in transforms
        return self.backbone(x)  # (B, out_dim)


class ProjectionHead(nn.Module):
    """2-layer MLP projection head"""

    def __init__(self, in_dim, hidden_dim=256, out_dim=256):  # fix: hidden_dim default → 256
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MOONModel(nn.Module):

    def __init__(self, encoder, proj_dim=256, num_classes=10):
        super().__init__()
        self.encoder = encoder
        enc_dim = encoder.out_dim          # 192 for deit_tiny
        self.proj_head = ProjectionHead(enc_dim, hidden_dim=256, out_dim=proj_dim)  # fix: explicit args
        self.classifier = nn.Linear(enc_dim, num_classes)

    def forward(self, x):
        rep = self.encoder(x)
        return self.classifier(rep), self.proj_head(rep)


def MOON_contrastive_loss(z, z_glob, z_prev, temperature=0.5):
    """
    l_con = -log[ exp(sim(z, z_glob)/τ) /
                 (exp(sim(z, z_glob)/τ) + exp(sim(z, z_prev)/τ)) ]

    Increases similarity between current local (z) and global (z_glob),
    while decreasing similarity between current local (z) and previous local (z_prev).
    """

    # fix: normalize before similarity — standard contrastive learning practice
    z      = F.normalize(z,      dim=-1)
    z_glob = F.normalize(z_glob, dim=-1)
    z_prev = F.normalize(z_prev, dim=-1)

    pos_sim = sim(z, z_glob, dim=-1) / temperature
    neg_sim = sim(z, z_prev, dim=-1) / temperature

    # Stack into logits: [pos, neg] per sample
    logits = torch.stack([pos_sim, neg_sim], dim=1)  # fix: stack instead of cat+reshape

    # Label 0 = maximise pos_sim (similarity to global model)
    labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)  # fix: explicit dtype + device

    # cross_entropy = softmax + log loss combined
    return F.cross_entropy(logits, labels)