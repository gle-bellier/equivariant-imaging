import torch
import torch.nn as nn

from equivariant_imaging.models.conv_block import ConvBlock


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, norm):
        super(UpBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels * 2,
                               out_channels,
                               dilation=dilation,
                               norm=norm)
        self.conv2 = ConvBlock(out_channels,
                               out_channels,
                               dilation=1,
                               norm=norm)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, ctx):

        x = torch.cat((self.up(x), ctx), 1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x