import random

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos, linspace, pi
from scipy.integrate import odeint, solve_bvp, solve_ivp
import numpy as np



if __name__ == '__main__':
    t1 = torch.randn(20,10)
    output = torch.sum(t1, dim=1)

