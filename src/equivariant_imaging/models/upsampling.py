import torch
import torch.nn as nn

from equivariant_imaging.models.conv_block import ConvBlock


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(UpBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, dilation=dilation)
        self.conv2 = ConvBlock(out_channels, out_channels, dilation=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, ctx):

        x = torch.cat((x, ctx), 1)

        x = self.conv1(x)
        x = self.conv2(x)
        return self.up(x)