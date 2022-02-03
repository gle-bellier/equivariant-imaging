import torch
import torch.nn as nn
from typing import List, Tuple, OrderedDict

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from equivariant_imaging.models.unet import Unet
from equivariant_imaging.physics.cs import CS


class EI(pl.LightningModule):
    def __init__(self, g_down_channels: List[int], g_up_channels: List[int],
                 g_down_dilations: List[int], g_up_dilations: List[int],
                 criteron: float, lr: float):
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

        self.G = Unet(down_channels=g_down_channels,
                      up_channels=g_up_channels,
                      down_dilations=g_down_dilations,
                      up_dilations=g_up_dilations)

        self.criteron = criteron
        self.val_idx = 0

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute pass forward
        """
        return self.G(self.cs.A_dagger(y))

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
        # TODO : get image from batch
        x = batch[0]

        # compute compressed version of x
        y = self.cs.A(x)

        # training routine
        x1 = self(y)

        pass

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Compute validation step (do some logging)
        Args:
            batch (torch.Tensor): batch
            batch_idx (int): batch index
        """
        self.val_idx += 1
        if self.val_idx % 10 == 0:
            pass

    def configure_optimizers(self) -> Tuple:
        """Configure both generator and discriminator optimizers
        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        opt = torch.optim.Adam(self.gen.parameters(),
                               lr=self.hparams.lr,
                               betas=(0.5, 0.999))

        return opt


if __name__ == "__main__":
    # get dataset

    lr = 1e-3
    # init model
    model = EI(g_down_channels=[2, 32, 64, 128],
               g_up_channels=[512, 128, 64, 32, 2],
               g_down_dilations=[3, 1, 1, 1],
               g_up_dilations=[3, 1, 1, 1, 1],
               lr=lr)

    trainer = pl.Trainer(gpus=1, max_epochs=10000)

    trainer.fit(model)