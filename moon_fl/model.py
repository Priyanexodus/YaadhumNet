import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity as sim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity as sim


class CNNEncoder(nn.Module):
    """Base encoder from MOON paper (Section 4.1) — ~62K params, ~5MB"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.out_dim = 84  # matches ViTEncoder interface

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc2(F.relu(self.fc1(x))))  # (B, 84)


class ProjectionHead(nn.Module):
    """2-layer MLP projection head"""
    def __init__(self, in_dim, hidden_dim=256, out_dim=256):
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
        self.encoder   = encoder
        enc_dim        = encoder.out_dim
        self.proj_head = ProjectionHead(enc_dim, hidden_dim=128, out_dim=proj_dim)
        self.classifier = nn.Linear(enc_dim, num_classes)

    def forward(self, x):
        rep = self.encoder(x)
        return self.classifier(rep), self.proj_head(rep)


def MOON_contrastive_loss(z, z_glob, z_prev, temperature=0.5):
    z      = F.normalize(z,      dim=-1)
    z_glob = F.normalize(z_glob, dim=-1)
    z_prev = F.normalize(z_prev, dim=-1)

    pos_sim = sim(z, z_glob, dim=-1) / temperature
    neg_sim = sim(z, z_prev, dim=-1) / temperature

    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)

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