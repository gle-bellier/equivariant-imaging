import torch
import torch.nn as nn

from equivariant_imaging.models.conv_block import ConvBlock


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, dilation=dilation)
        self.conv2 = ConvBlock(out_channels, out_channels, dilation=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x
