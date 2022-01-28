import torch
import torch.nn as nn

from equivariant_imaging.models.downsampling import DownBlock
from equivariant_imaging.models.upsampling import UpBlock
from equivariant_imaging.models.conv_block import ConvBlock


class Unet(nn.Module):
    def __init__(self, down_channels, up_channels, down_dilations,
                 up_dilations):
        super(Unet, self).__init__()

        self.down_blocks = nn.ModuleList([
            DownBlock(in_c, out_c, dilation) for in_c, out_c, dilation in zip(
                down_channels[:-1], down_channels[1:], down_dilations)
        ])

        self.up_blocks = nn.ModuleList([
            UpBlock(in_c, out_c, dilation) for in_c, out_c, dilation in zip(
                up_channels[:-1], up_channels[1:], up_dilations)
        ])
