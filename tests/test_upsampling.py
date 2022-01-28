import torch
import torch.nn as nn

from equivariant_imaging.models.upsampling import UpBlock


def test_upsampling():
    batch_size = 10
    kernel_size = 3
    in_channels = 64
    w, l = 32, 32
    out_channels = 32

    in_c = torch.randn((batch_size, out_channels, w, l))
    ctx = torch.randn((batch_size, out_channels, w, l))

    assert UpBlock(in_channels, out_channels,
                   dilation=1)(in_c, ctx).shape == (batch_size, out_channels,
                                                    w * 2, l * 2)
