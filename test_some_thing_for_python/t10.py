

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import jax.random as random
import torch

from DeepONet_ode.utils import to_tensor

if __name__ == '__main__':
    t1 = np.random.randn(3,1)
    t2 = np.random.randn(3, 1)

    # res = np.stack([t1,t2])z
    t = t1 - t2
    t = to_tensor(t)
    t3 = torch.zeros_like(t)


