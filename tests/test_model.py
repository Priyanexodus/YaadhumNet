import numpy as np
import pytest, torch
from src.model import MOONModel, ViTEncoder, MOON_contrastive_loss

def test_vit_encoder_output_shape():
    enc = ViTEncoder('deit_tiny_patch16_224', pretrained=False)
    model = MOONModel(enc, proj_dim=256, num_classes=10)
    x = torch.randn(4, 3, 224, 224) # batch=4
    logits, z = model(x)
    assert logits.shape == (4, 10), f'Classifier output: {logits.shape}'
    assert z.shape == (4, 256), f'Projection output: {z.shape}'

def test_moon_contrastive_loss_positive_alignment():
    """Loss should be lower when local is closer to global than to prev"""
    B,D = 8, 256
    z_glob = torch.randn(B, D); z_glob = z_glob / z_glob.norm(dim=-1,
    keepdim=True)
    z_close = z_glob + 0.01 * torch.randn(B, D) # very close to global
    z_far = torch.randn(B, D) # random previous
    loss_good = MOON_contrastive_loss(z_close, z_glob, z_far)
    loss_bad = MOON_contrastive_loss(z_far, z_glob, z_close)
    assert loss_good < loss_bad, 'Loss should reward proximity to global model'

def test_parameter_serialization_roundtrip():
    enc = ViTEncoder('deit_tiny_patch16_224', pretrained=False)
    model = MOONModel(enc, proj_dim=256, num_classes=10)
    params = [v.cpu().numpy() for v in model.state_dict().values()]
    keys = list(model.state_dict().keys())
    state = dict(zip(keys, [torch.tensor(p) for p in params]))
    model.load_state_dict(state, strict=True) # should not raise