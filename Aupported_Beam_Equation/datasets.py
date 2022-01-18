from torch.utils.data import Dataset

from pyDOE import lhs
import numpy as np
import scipy.io
import torch


class InitialConditionDataset:
    def __init__(self, N_u, N_f, lb, ub):
        """
        Constructor of the initial condition dataset

        Args:
          N_u (int)
          N_f (int)
          lb (List)
          ub (List)
        """
        super(type(self)).__init__()

        # data ----------------------------------------------------------------
        # data = scipy.io.loadmat("Aupported_Beam_Equation/data/burgers_shock.mat")

        t = np.linspace(0, 2, 256).flatten()[:, None] 
        x = np.linspace(0, 1, 256).flatten()[:, None]
        Exact = np.zeros([256, 256])  # Exact （t，x）

        X, T = np.meshgrid(x, t)  # X(100,256) T(100,256)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # (25600, 2)

        # Doman bounds----------------------------------------------------------------

        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))  # 左
        uu1 = np.zeros([256, 1])
        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))  # 下
        uu2 = np.zeros([256, 1])
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))  # 上
        uu3 = np.zeros([256, 1])

        # all bounds constaints
        X_u_train = np.vstack([xx1, xx2, xx3])
        u_train = np.vstack([uu1, uu2, uu3])

        # pde constaints
        X_f_train = lb + (ub - lb) * lhs(2, N_f)
        X_f_train = np.vstack((X_f_train, X_u_train))

        # ib constraints
        idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
        X_u_train = X_u_train[idx, :]
        u_train = u_train[idx, :]

        # u_xx constraints
        X_u_derivative_xx_trian = np.vstack([xx2, xx3])
        u_derivative_xx_trian = np.vstack([uu2, uu3])
        idx = np.random.choice(
            X_u_derivative_xx_trian.shape[0], int(N_u / 4), replace=False
        )
        X_u_derivative_xx_trian = X_u_derivative_xx_trian[idx, :]
        u_derivative_xx_trian = u_derivative_xx_trian[idx, :]

        # u_t constraints
        idx = np.random.choice(xx1.shape[0], int(N_u /2), replace=False)
        X_u_derivative_t_trian = xx1[idx, :]
        u_derivative_t_trian = (
            0.04 * X_u_derivative_t_trian * (1 - X_u_derivative_t_trian)
        )[:, 0][:, None]

        self.X_star = X_star
        self.X_f_train = X_f_train
        self.X_u_train = X_u_train
        self.u_train = u_train
        self.X_u_derivative_xx_trian = X_u_derivative_xx_trian
        self.u_derivative_xx_trian = u_derivative_xx_trian
        self.X_u_derivative_t_trian = X_u_derivative_t_trian
        self.u_derivative_t_trian = u_derivative_t_trian

    def getData(self):
        return (
            self.X_star,
            self.X_f_train,
            self.X_u_train,
            self.u_train,
            self.X_u_derivative_xx_trian,
            self.u_derivative_xx_trian,
            self.X_u_derivative_t_trian,
            self.u_derivative_t_trian,
        )
