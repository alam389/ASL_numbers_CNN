"""Training-time augmentation for hand crops (tensors in [-1, 1], shape (1, H, W)).

Always: random affine and brightness/contrast. Optional background swap (p_background > 0)
is only for black-background datasets: it finds the hand with a brightness threshold and
pastes it onto random synthetic backgrounds so the model cannot rely on "bright shape on black".
"""
import math

import torch
import torch.nn.functional as F
import torchvision.transforms as T

BLACK = -1.0
HAND_THRESHOLD = BLACK + 0.08      # darker than this counts as background


def hand_mask(x):
    """Soft-ish binary mask of the hand (1 = hand), slightly dilated to keep shadows."""
    m = (x > HAND_THRESHOLD).float().unsqueeze(0)
    m = F.max_pool2d(m, 3, stride=1, padding=1)    # dilate
    m = -F.max_pool2d(-m, 3, stride=1, padding=1)  # close small holes
    return m[0]


def random_background(shape):
    """A random gray background in [-1, 1]: flat, gradient or blotchy, maybe with a light stripe."""
    _, H, W = shape
    base = torch.rand(1).item() * 1.6 - 0.8
    kind = torch.randint(0, 3, (1,)).item()
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    if kind == 0:
        bg = torch.full(shape, base)
    elif kind == 1:
        angle = torch.rand(1).item() * 2 * math.pi
        ramp = (xs * math.cos(angle) + ys * math.sin(angle)).unsqueeze(0)
        bg = base + ramp * (torch.rand(1).item() * 0.6)
    else:
        coarse = torch.randn(1, 1, 6, 6)
        bg = base + F.interpolate(coarse, size=(H, W), mode="bicubic", align_corners=False)[0] * 0.35
    if torch.rand(1).item() < 0.4:                  # bright/dark stripe, like a ceiling light
        angle = torch.rand(1).item() * math.pi
        dist = (xs * math.sin(angle) - ys * math.cos(angle) - (torch.rand(1).item() - 0.5)).abs()
        stripe = (dist < 0.05 + torch.rand(1).item() * 0.1).float().unsqueeze(0)
        bg = bg * (1 - stripe) + stripe * (torch.rand(1).item() * 2 - 1)
    bg = bg + torch.randn(shape) * 0.03
    return bg.clamp(-1, 1)


class HandAugment:
    """affine -> random background (with prob p_background) -> brightness/contrast."""

    def __init__(self, p_background=0.0):
        self.p_background = p_background
        self.affine = T.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.85, 1.15), fill=BLACK)

    def __call__(self, x):
        x = self.affine(x)
        if torch.rand(1).item() < self.p_background:
            m = hand_mask(x)
            x = x * m + random_background(x.shape) * (1 - m)
        x = x * (0.8 + 0.4 * torch.rand(1)) + (torch.rand(1) - 0.5) * 0.4
        return x.clamp(-1, 1)
