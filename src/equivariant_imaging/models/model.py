import torch
import torch.nn as nn
from typing import List, Tuple, OrderedDict

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import os

from equivariant_imaging.models.unet import Unet
from equivariant_imaging.physics.cs import CS
from equivariant_imaging.transforms.random_shift import Shift


class EI(pl.LightningModule):
    def __init__(self,
                 g_down_channels: List[int],
                 g_up_channels: List[int],
                 g_down_dilations: List[int],
                 g_up_dilations: List[int],
                 criteron: float,
                 lr: float,
                 alpha=0.5):
        """[summary]
        Args:
            g_down_channels (List[int]): generator list of downsampling channels
            g_up_channels (List[int]): generator list of upsampling channels
            g_down_dilations (List[int]): generator list of down blocks dilations
            g_up_dilations (List[int]): generator list of up blocks dilations
            criteron (float): criteron for both generator and discriminator
            lr (float): learning rate
        """
        super(EI, self).__init__()

        self.save_hyperparameters()

        # TODO : choose correct d and D and image shape
        # instantiate compressed sensing
        self.cs = CS(64, 256, [1, 16, 16])
        # instantiate tranformation
        self.T = Shift(n_trans=2)

        self.G = Unet(down_channels=g_down_channels,
                      up_channels=g_up_channels,
                      down_dilations=g_down_dilations,
                      up_dilations=g_up_dilations)

        self.f = lambda y: self.G(self.cs.A_dagger(y))

        self.criteron = criteron
        self.val_idx = 0

        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> Tuple(torch.Tensor):
        """
        Compute pass forward
        """
        y = self.cs.A(x)

        # training routine
        x1 = self.f(y)
        x2 = self.T.apply(x1)
        x3 = self.f(self.cs.A(x))

        return y, x1, x2, x3

    def __loss(self, y, x1, x2, x3):
        return torch.nn.MSE(self.cs.A(x1),
                            y) + self.alpha * torch.nn.MSE(x3, x2)

    def training_step(self, batch: List[torch.Tensor],
                      batch_idx: int) -> OrderedDict:
        """Compute a training step for generator or discriminator 
        (according to optimizer index)
        Args:
            batch (List[torch.Tensor]): batch 
            batch_idx (int): batch index
        Returns:
            OrderedDict: dict {loss, progress_bar, log}
        """

        y, x1, x2, x3 = self(batch)
        return self.__loss(y, x1, x2, x3)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Compute validation step (do some logging)
        Args:
            batch (torch.Tensor): batch
            batch_idx (int): batch index
        """

        y, x1, x2, x3 = self(batch)
        return self.__loss(y, x1, x2, x3)

    def configure_optimizers(self) -> Tuple:
        """Configure both generator and discriminator optimizers
        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        opt = torch.optim.Adam(self.gen.parameters(),
                               lr=self.hparams.lr,
                               betas=(0.5, 0.999))

        return opt

    def train_dataloader(self):
        # transforms
        # prepare transforms standard to MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        # data
        mnist_train = MNIST('./data/',
                            train=True,
                            download=True,
                            transform=transform)
        return DataLoader(mnist_train, batch_size=64)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        mnist_val = MNIST('./data/',
                          train=False,
                          download=True,
                          transform=transform)
        return DataLoader(mnist_val, batch_size=64)


if __name__ == "__main__":
    # get dataset

    lr = 1e-3
    # init model
    model = EI(g_down_channels=[2, 32, 64, 128],
               g_up_channels=[512, 128, 64, 32, 2],
               g_down_dilations=[3, 1, 1, 1],
               g_up_dilations=[3, 1, 1, 1, 1],
               lr=lr,
               alpha=0.5)

    trainer = pl.Trainer(gpus=1, max_epochs=10000)

    trainer.fit(model)