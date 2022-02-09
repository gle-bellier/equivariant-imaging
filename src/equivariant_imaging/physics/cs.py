import torch
import os
import numpy as np

rng = np.random.default_rng(123)

# source : https://colab.research.google.com/github/edongdongchen/EI/blob/main/ei_demo_cs_usps.ipynb#scrollTo=razUOZiJy_ip


# define an inverse problem e.g. Compressed Sensing (CS)
# where the forward operator A is a random projection matrix
class CS():
    def __init__(self, d, D, img_shape, dtype=torch.float):
        self.img_shape = img_shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        fname = 'data/matrices/cs_{}x{}.pt'.format(d, D)
        if os.path.exists(fname):
            A, A_dagger = torch.load(fname)
        else:
            A = rng.randn(d, D) / np.sqrt(d)
            A_dagger = np.linalg.pinv(A)
            torch.save([A, A_dagger], fname)
            print('CS matrix has been created and saved at {}'.format(fname))
        self._A = torch.from_numpy(A).float().to(device)
        self._A_dagger = torch.from_numpy(A_dagger).float().to(device)

    def A(self, x):

        y = torch.einsum('in, mn->im', x.reshape(x.shape[0], -1), self._A)
        return y

    def A_dagger(self, y):
        N, C, H, W = y.shape[0], self.img_shape[0], self.img_shape[
            1], self.img_shape[2]
        x = torch.einsum('im, nm->in', y, self._A_dagger)
        x = x.reshape(N, C, H, W)
        return x
