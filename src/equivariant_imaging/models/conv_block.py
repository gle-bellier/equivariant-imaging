import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 stride: int,
                 dilation: int,
                 norm=False,
                 dropout=0.) -> None:
        """Create 2D Convolutional block composed of a convolutional layer
        followed by batch normalization and leaky ReLU.
        Args:
            in_channels (int): input number of channels
            out_channels (int): output number of channels
            kernel size (int): kernel size of the convolutional layer
            padding (int): padding of the convolutional layer
            stride (int): stride of the convolutional layer
            dilation (int): dilation of the convolutional layer
            norm (bool, optional): process batchnorm. Defaults to False.
            dropout (float, optional): dropout probability. Defaults to 0.
        """

        super().__init__()
        self.norm = norm
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              padding=padding,
                              stride=stride)

        self.lr = nn.LeakyReLU(.2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass 
        Args:
            x (torch.Tensor): input contours
        Returns:
            torch.Tensor: output contours
        
        """
        x = self.dp(x)
        x = self.conv(x)
        out = self.lr(x)

        if self.norm:
            out = self.bn(out)

        return out