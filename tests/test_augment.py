import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augment import HandAugment, hand_mask, random_background


def _hand_on_black():
    x = torch.full((1, 128, 128), -1.0)
    x[:, 40:90, 50:80] = 0.4
    return x


def test_mask_covers_hand_not_corners():
    m = hand_mask(_hand_on_black())
    assert m[0, 60, 60] == 1 and m[0, 0, 0] == 0


def test_background_shape_and_range():
    for _ in range(20):
        bg = random_background((1, 128, 128))
        assert bg.shape == (1, 128, 128) and bg.min() >= -1 and bg.max() <= 1


def test_augment_keeps_shape_range_and_changes_background():
    aug = HandAugment(p_background=1.0)
    out = aug(_hand_on_black())
    assert out.shape == (1, 128, 128) and out.min() >= -1 and out.max() <= 1
    # corners were black; after the background swap they should (almost always) differ
    torch.manual_seed(0)
    corners = torch.stack([aug(_hand_on_black())[0, :10, :10].mean() for _ in range(10)])
    assert (corners > -0.95).any()
