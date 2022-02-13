

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import jax.random as random
import torch

from DeepONet_ode.utils import to_tensor


if __name__ == '__main__':
    def square(x,y):  # 计算平方数
        return x ** 2+y

    input1 = np.array([1,2,3])
    y_tile = np.tile(1,input1.shape)
    res = list(map(square, input1,y_tile))
    print(res  )

