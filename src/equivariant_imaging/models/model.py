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
                 comp_ratio: int,
                 lr: float,
                 norm=False,
                 alpha=1.,
                 batch_size=64):
        """Equivariant-imaging class (model and data and training routine)

        Args:
            g_down_channels (List[int]): generator list of downsampling channels
            g_up_channels (List[int]): generator list of upsampling channels
            g_down_dilations (List[int]): generator list of down blocks dilations
            g_up_dilations (List[int]): generator list of up blocks dilations
            comp_factor (int): compression factor (size of A ~ size image / comp_factor) (impacts size of A)
            lr (float): learning rate
            norm (bool, optional): [description]. batchnorm to False.
            alpha (float, optional): [description]. ratio between pseudo-inverse learning loss and the equivariance loss to 1.
            batch_size (int, optional): batch size. Defaults to 64.
        """
        super(EI, self).__init__()

        self.save_hyperparameters()

        # TODO : choose correct d and D and image shape

        self.G = Unet(down_channels=g_down_channels,
                      up_channels=g_up_channels,
                      down_dilations=g_down_dilations,
                      up_dilations=g_up_dilations,
                      norm=norm)

        # instantiate compressed sensing

        self.image_size = 32
        self.comp_ratio = comp_ratio
        self.cs = CS((self.image_size // self.comp_ratio)**2,
                     self.image_size**2, [1, self.image_size, self.image_size])
        # instantiate tranformation
        self.T = Shift(n_trans=2)

        self.f = lambda y: self.G(self.cs.A_dagger(y))

        self.val_idx = 0
        self.train_idx = 0

        self.alpha = alpha
        self.batch_size = batch_size

        #Transform to resize the image and normalize
        self.transform = transforms.Compose([
            transforms.Pad(2, padding_mode="edge"),
            transforms.ToTensor(),
        ])

        self.invtransform = transforms.Compose([transforms.CenterCrop(28)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Compute pass forward
        """
        y = self.cs.A(x)

        # training routine

        x1 = self.f(y)
        x2 = self.T.apply(x1)
        x3 = self.f(self.cs.A(x2))

        return y, x1, x2, x3

    def __loss(self, y: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
               x3: torch.Tensor) -> Tuple[torch.Tensor]:
        """Compute the loss function (does not include the GAN loss)
        """

        pinv_loss = torch.nn.functional.mse_loss(self.cs.A(x1), y)
        ei_loss = torch.nn.functional.mse_loss(x3, x2)

        return pinv_loss, ei_loss, pinv_loss + self.alpha * ei_loss

    def __PSNR(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Peak Signal Noise Ratio to evaluate the quality 
        of the reconstruction

        Args:
            x (torch.Tensor): original image
            y (torch.Tensor): reconstructed image

        Returns:
            torch.Tensor: PSNR
        """
        # dynamic of the signal : in our case max of the image : 1.
        d = 1.
        return 10 * torch.log10(d / nn.functional.mse_loss(x, y))

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

        pinv_loss, ei_loss, loss = self.__loss(y, x1, x2, x3)
        psnr = self.__PSNR(x, x1)

        self.train_idx += 1
        if self.train_idx % 100 == 0:
            self.log("train/PSNR", psnr)
            self.log("train/pinv_loss", pinv_loss)
            self.log("train/ei_loss", ei_loss)
            self.log("train/train_loss", loss)
            self.logger.experiment.add_image("train/original",
                                             self.invtransform(x[0]),
                                             self.val_idx)
            self.logger.experiment.add_image("train/reconstruct",
                                             self.invtransform(x1[0]),
                                             self.val_idx)

        return dict(loss=loss, log=dict(train_loss=loss.detach()))

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Compute validation step (do some logging)
        Args:
            batch (torch.Tensor): batch
            batch_idx (int): batch index
        """
        x, label = batch
        y, x1, x2, x3 = self(x)

        # compute reconstruction only with pseudo inverse
        pinv_rec = self.cs.A_dagger(y)

        pinv_loss, ei_loss, loss = self.__loss(y, x1, x2, x3)
        psnr = self.__PSNR(x, x1)

        self.val_idx += 1
        if self.val_idx % 100 == 0:
            self.log("valid/PSNR", psnr)
            self.log("valid/pinv_loss", pinv_loss)
            self.log("valid/ei_loss", ei_loss)
            self.log("valid/val_loss", loss)
            self.logger.experiment.add_image("valid/original",
                                             self.invtransform(x[0]),
                                             self.val_idx)
            self.logger.experiment.add_image("valid/pinv",
                                             self.invtransform(pinv_rec[0]),
                                             self.val_idx)
            self.logger.experiment.add_image("valid/reconstruct",
                                             self.invtransform(x1[0]),
                                             self.val_idx)

        return dict(validation_loss=loss, log=dict(val_loss=loss.detach()))

    def configure_optimizers(self) -> Tuple:
        """Configure both generator and discriminator optimizers
        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        opt = torch.optim.Adam(self.G.parameters(), lr=self.hparams.lr)
        sc = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                        verbose=True,
                                                        patience=4,
                                                        factor=0.5)

        return {
            'optimizer': opt,
            'lr_scheduler': sc,
            "monitor": "train/train_loss"
        }

    def train_dataloader(self):

        mnist_train = MNIST('./data/',
                            train=True,
                            download=True,
                            transform=self.transform)
        return DataLoader(mnist_train,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):

        mnist_val = MNIST('./data/',
                          train=False,
                          download=True,
                          transform=self.transform)
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
