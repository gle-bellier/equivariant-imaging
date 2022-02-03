import torch
import torch.nn as nn

from equivariant_imaging.models.conv_block import ConvBlock


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, norm):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels,
                               out_channels,
                               dilation=dilation,
                               norm=norm)
        self.conv2 = ConvBlock(out_channels,
                               out_channels,
                               dilation=1,
                               norm=norm)
        self.mp = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        return self.mp(x), x
