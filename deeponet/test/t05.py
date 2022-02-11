import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos, linspace, pi
from scipy.integrate import odeint, solve_bvp, solve_ivp
import numpy as np



if __name__ == '__main__':
    N =10
    batch_size=20
    idx = np.random.choice(N, (batch_size,), replace=False)
    print(idx)

