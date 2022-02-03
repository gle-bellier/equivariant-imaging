import torch
import torch.nn as nn

from equivariant_imaging.models.downsampling import DownBlock
from equivariant_imaging.models.upsampling import UpBlock
from equivariant_imaging.models.bottleneck import Bottleneck

from equivariant_imaging.models.conv_block import ConvBlock


class Unet(nn.Module):
    def __init__(self,
                 down_channels,
                 up_channels,
                 down_dilations,
                 up_dilations,
                 norm=False):
        super(Unet, self).__init__()

        self.down_blocks = nn.ModuleList([
            DownBlock(in_c, out_c, dilation, norm=norm)
            for in_c, out_c, dilation in zip(down_channels[:-1],
                                             down_channels[1:], down_dilations)
        ])

        self.up_blocks = nn.ModuleList([
            UpBlock(in_c, out_c, dilation,
                    norm=norm) for in_c, out_c, dilation in zip(
                        up_channels[:-1], up_channels[1:], up_dilations)
        ])

        self.bottleneck = Bottleneck(down_channels[-1], down_channels[-1])

        self.top = nn.Sequential(
            nn.Conv2d(up_channels[-1],
                      up_channels[-1],
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.Tanh())

    def forward(self, x):

        l_ctx = []
        for d_block in self.down_blocks:
            x, ctx = d_block(x)
            #print(x.shape, ctx.shape)
            l_ctx += [ctx]

        x = self.bottleneck(x)

        for i, u_block in enumerate(self.up_blocks):
            #print(x.shape, l_ctx[-i - 1].shape)
            x = u_block(x, l_ctx[-i - 1])
        x = self.top(x)
        return x
