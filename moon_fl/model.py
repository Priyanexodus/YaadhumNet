import copy, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.functional import cosine_similarity as sim
import torch.optim as optim
import timm

class ViTEncoder(nn.Module):

    """
    Lightweight ViT encoder using timm.
    deit_tiny_patch16_224 gives strong features at low compute cost.
    For heavier compute: vit_small_patch16_224 or vit_base_patch16_224
    """

    def __init__(self, model_name='deit_tiny_patch16_224',pretrained=True, out_dim=192):

        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, # remove classification head
            global_pool='avg' # average pool CLS token
        )
        self.out_dim = self.backbone.num_features

    def forward(self, x):
        # timm expects 224x224; add resize in transforms
        return self.backbone(x) # (B, out_dim)

class ProjectionHead(nn.Module):
    """2-layer MLP projection head"""
    def __init__(self, in_dim, hidden_dim=128, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class MOONModel(nn.Module):
    def __init__(self, encoder, proj_dim=256, num_classes=10):

        super().__init__()
        self.encoder = encoder
        enc_dim = encoder.out_dim # 192 for deit_tiny
        self.proj_head = ProjectionHead(enc_dim, 256, proj_dim)
        self.classifier = nn.Linear(enc_dim, num_classes)

    def forward(self, x):
        rep = self.encoder(x)
        return self.classifier(rep), self.proj_head(rep)


def MOON_contrastive_loss(z, z_glob, z_prev, temperature=0.5):
    """
    l_con = -log[ exp(sim(z, z_glob)/τ) /
                    (exp(sim(z, z_glob)/τ) + exp(sim(z, z_prev)/τ)) ]

    pos_sim represents the similarity between the global projection and local (current) projection.
    neg_sim represents the similarity between the local (current) projection and local (previous) projection.

    We have to increase the similarity between the global projection and local (current) projection,
    decrease the similarity between the local (current) projection and local (previous) projection.
    """

    pos_sim = sim(z, z_glob, dim=-1) / temperature
    neg_sim = sim(z, z_prev, dim=-1) / temperature

    logits = torch.cat([pos_sim.reshape(-1,1), neg_sim.reshape(-1,1)], dim=1)

    #The labels is used to selected to maximise the pos_sim by guiding it's in the zeroth position.
    labels = torch.zeros(z.size(0)).long().to(z.device)

    #Since the cross entropy loss consist of both the softmax and logloss it is directly employed here to reduce the boilerplate code.
    con_loss = F.cross_entropy(logits, labels)

    return con_loss