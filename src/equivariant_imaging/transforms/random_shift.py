# define a transformation group (random shift)
import random
import torch
import numpy as np


class Shift():
    def __init__(self, n_trans):
        self.n_trans = n_trans

    def apply(self, x):
        H, W = x.shape[-2], x.shape[-1]
        assert self.n_trans <= H - 1 and self.n_trans <= W - 1, 'n_shifts should less than {}'.format(
            H - 1)

        shifts_row = random.sample(
            list(np.concatenate([-1 * np.arange(1, H),
                                 np.arange(1, H)])), self.n_trans)
        shifts_col = random.sample(
            list(np.concatenate([-1 * np.arange(1, W),
                                 np.arange(1, W)])), self.n_trans)

        x = torch.cat([
            x if self.n_trans == 0 else torch.roll(
                x, shifts=[sx, sy], dims=[-2, -1]).type_as(x)
            for sx, sy in zip(shifts_row, shifts_col)
        ],
                      dim=0)
        return x
