import torch
import torch.nn as nn

from equivariant_imaging.models.downsampling import DownBlock


def test_downsampling():
    batch_size = 10
    kernel_size = 3
    in_channels = 32
    w, l = 32, 32
    out_channels = 64

    in_c = torch.randn((batch_size, in_channels, w, l))

    assert DownBlock(in_channels, out_channels,
                     dilation=1)(in_c).shape == (batch_size, out_channels,
                                                 w // 2, l // 2)
