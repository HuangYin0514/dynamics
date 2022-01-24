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
        n_t = 100
        n_x = 256
        t = np.linspace(lb[1], ub[1], n_t).flatten()[:, None]
        x = np.linspace(lb[0], ub[0], n_x).flatten()[:, None]
        Exact = np.zeros([n_t, n_x])  # Exact （t，x）

        X, T = np.meshgrid(x, t)  # X(n_t,n_x) T(n_t,n_x)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # (n_x*n_t, 2)

        # Doman bounds----------------------------------------------------------------

        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))  # 左
        uu1 = -1 * np.sin(xx1 * np.pi)[:, 0][:, None]
        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))  # 下
        uu2 = np.zeros([n_t, 1])
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))  # 上
        uu3 = np.zeros([n_t, 1])

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

        self.X_star = X_star
        self.X_f_train = X_f_train
        self.X_u_train = X_u_train
        self.u_train = u_train

    def getData(self):
        return (
            self.X_star,
            self.X_f_train,
            self.X_u_train,
            self.u_train,
        )
