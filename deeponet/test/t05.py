import random

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos, linspace, pi
from scipy.integrate import odeint, solve_bvp, solve_ivp
import numpy as np
from torch import nn

if __name__ == '__main__':
    loss = nn.MSELoss()
    t1 = torch.randn(20,)
    t2 = torch.randn(20, 1)
    loss_res = loss(t1.flatten(),t2)
    print(loss_res)

