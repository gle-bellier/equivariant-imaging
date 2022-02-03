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
                 lr: float,
                 alpha=0.5,
                 batch_size=64):
        """[summary]
        Args:
            g_down_channels (List[int]): generator list of downsampling channels
            g_up_channels (List[int]): generator list of upsampling channels
            g_down_dilations (List[int]): generator list of down blocks dilations
            g_up_dilations (List[int]): generator list of up blocks dilations
            lr (float): learning rate
        """
        super(EI, self).__init__()

        self.save_hyperparameters()

        # TODO : choose correct d and D and image shape

        self.G = Unet(down_channels=g_down_channels,
                      up_channels=g_up_channels,
                      down_dilations=g_down_dilations,
                      up_dilations=g_up_dilations)

        # instantiate compressed sensing

        self.cs = CS(64, 32**2, [1, 32, 32])
        # instantiate tranformation
        self.T = Shift(n_trans=2)

        self.f = lambda y: self.G(self.cs.A_dagger(y))

        self.val_idx = 0

        self.alpha = alpha

        self.batch_size = batch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Compute pass forward
        """
        y = self.cs.A(x)

        # training routine

        x1 = self.f(y)
        x2 = self.T.apply(x1)
        x3 = self.f(self.cs.A(x2))

        # print(
        #     f"y : {y.shape}\n x1 : {x1.shape}\n x1 Transformed: {self.cs.A(x1).shape}\n x2 : {x2.shape}\n x3 : {x3.shape}\n"
        # )
        # input()

        return y, x1, x2, x3

    def __loss(self, y, x1, x2, x3):

        return torch.nn.functional.mse_loss(
            self.cs.A(x1),
            y) + self.alpha * torch.nn.functional.mse_loss(x3, x2)

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

        x, label = batch
        y, x1, x2, x3 = self(x)
        loss = self.__loss(y, x1, x2, x3)

        self.log("train_loss", loss)
        return dict(loss=loss, log=dict(train_loss=loss.detach()))

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Compute validation step (do some logging)
        Args:
            batch (torch.Tensor): batch
            batch_idx (int): batch index
        """
        x, label = batch
        y, x1, x2, x3 = self(x)

        loss = self.__loss(y, x1, x2, x3)

        # plot some images
        self.log("val_loss", loss)
        self.logger.experiment.add_image("original", x[0], self.val_idx)
        self.logger.experiment.add_image("reconstruct", x1[0], self.val_idx)
        self.val_idx += 1

        return dict(validation_loss=loss, log=dict(val_loss=loss.detach()))

    def configure_optimizers(self) -> Tuple:
        """Configure both generator and discriminator optimizers
        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        opt = torch.optim.Adam(self.G.parameters(),
                               lr=self.hparams.lr,
                               betas=(0.5, 0.999))

        return opt

    def train_dataloader(self):
        # transforms
        # prepare transforms standard to MNIST
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        # data
        mnist_train = MNIST('./data/',
                            train=True,
                            download=True,
                            transform=transform)
        return DataLoader(mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        mnist_val = MNIST('./data/',
                          train=False,
                          download=True,
                          transform=transform)
        return DataLoader(mnist_val, batch_size=self.batch_size)


if __name__ == "__main__":
    # get dataset

    lr = 1e-3
    # init model
    model = EI(g_down_channels=[1, 2, 2, 4, 8],
               g_up_channels=[8, 4, 2, 2, 1],
               g_down_dilations=[1, 1, 1, 1],
               g_up_dilations=[1, 1, 1, 1],
               lr=lr,
               alpha=0.5,
               batch_size=8)

    trainer = pl.Trainer(gpus=0, max_epochs=10000)

    trainer.fit(model)
