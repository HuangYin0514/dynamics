import torch

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from model import *

np.random.seed(1234)

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    nu = 0.01 / np.pi
    noise = 0.0

    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 1]

    data = scipy.io.loadmat("Aupported_Beam_Equation/data/burgers_shock.mat")

    t = np.linspace(0, 1, 100).flatten()[:, None]
    x = np.linspace(0, 1, 256).flatten()[:, None]
    Exact = np.zeros([100,256])
    # Exact = np.real(data["usol"]).T

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Doman bounds
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 0.99])

    # 左 上 下 边界
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))  # 左
    uu1 = np.zeros([256, 1])
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))  # 下
    uu2 = np.zeros([100, 1])
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))  # 上
    uu3 = np.zeros([100, 1])

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    X_u_derivative_xx_trian = np.vstack([xx2, xx3])
    u_derivative_xx_trian = np.vstack([uu2, uu3])
    idx = np.random.choice(
        X_u_derivative_xx_trian.shape[0], int(N_u / 4), replace=False
    )
    X_u_derivative_xx_trian = X_u_derivative_xx_trian[idx, :]
    u_derivative_xx_trian = u_derivative_xx_trian[idx, :]

    idx = np.random.choice(xx1.shape[0], int(N_u / 4), replace=False)
    X_u_derivative_t_trian = xx1[idx, :]
    u_derivative_t_trian = uu1[idx, :]

    model = PhysicsInformedNN(
        X_u_train,
        u_train,
        X_u_derivative_xx_trian,
        u_derivative_xx_trian,
        X_u_derivative_t_trian,
        u_derivative_t_trian,
        X_f_train,
        layers,
        lb,
        ub,
        nu,
    )
    model.train()

    u_pred, f_pred = model.predict(X_star)
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")

    scipy.io.savemat("pred.mat", {'u_pred':u_pred})
    
    