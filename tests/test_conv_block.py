import pytest
import torch

from equivariant_imaging.models.conv_block import ConvBlock


def test_dims_conv1d():
    batch_size = 10
    in_channels = 32
    w, l = 32, 32
    out_channels = 64

    in_c = torch.randn((batch_size, in_channels, w, l))

    assert ConvBlock(in_channels, out_channels,
                     dilation=3)(in_c).shape == (batch_size, out_channels, w,
                                                 l)
