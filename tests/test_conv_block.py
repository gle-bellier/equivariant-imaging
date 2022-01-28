import pytest
import torch

from equivariant_imaging.models.conv_block import ConvBlock


def test_dims_conv2d():
    batch_size = 10

    kernel_size = 3
    in_channels = 32
    w, l = 32, 32
    out_channels = 64

    in_c = torch.randn((batch_size, in_channels, w, l))

    assert ConvBlock(in_channels,
                     out_channels,
                     kernel_size,
                     padding=3,
                     stride=1,
                     dilation=3)(in_c).shape == (batch_size, out_channels, w,
                                                 l)
