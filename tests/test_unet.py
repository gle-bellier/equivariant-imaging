import torch
import torch.nn as nn

from equivariant_imaging.models.unet import Unet


def test_downsampling():

    model = Unet([2, 4, 32, 256], [256, 32, 4, 2], [1, 1, 1], [1, 1, 1])
    x = torch.randn(10, 2, 32, 32)
    assert model(x).shape == x.shape
