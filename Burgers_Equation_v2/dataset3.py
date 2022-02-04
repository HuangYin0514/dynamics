import sys

import numpy as np
import scipy.io
from torch.utils.data import Dataset

try:
    from pyDOE import lhs
except ImportError:

    sys.path.append('/kaggle/input/pylib-pydoe/MySitePackages')
    from pyDOE import lhs


class BurgersEquationDataSet(Dataset):
    def __init__(self):
        # 采样点个数

        n_u = 100
        n_f = 10000
        data = scipy.io.loadmat('data/burgers_shock.mat')

        t = data['t'].flatten()[:, None]
        x = data['x'].flatten()[:, None]
        Exact = np.real(data['usol']).T

        X, T = np.meshgrid(x, t)

        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]

        # Doman bounds
        lb = X_star.min(0)
        ub = X_star.max(0)

        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
        uu1 = Exact[0:1, :].T
        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
        uu2 = Exact[:, 0:1]
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))
        uu3 = Exact[:, -1:]

        # all bounds constraints
        x_u_train = np.vstack([xx1, xx2, xx3])
        u_train = np.vstack([uu1, uu2, uu3])

        # pde constraints
        x_f_train = lb + (ub-lb)*lhs(2, n_f)
        self.x_f_train = np.vstack((x_f_train, x_u_train))

        # ib constraints
        idx = np.random.choice(x_u_train.shape[0], n_u, replace=False)
        self.x_u_train = x_u_train[idx, :]
        self.u_train = u_train[idx, :]

        # Prediction
        data = scipy.io.loadmat("data/burgers_shock.mat")
        exact = np.real(data["usol"]).T
        self.x_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # (n_x*n_t, 2)
        self.u_star = exact.flatten()[:, None]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x_f_train = self.x_f_train
        x_u_train = self.x_u_train
        u_train = self.u_train

        return [x_f_train, x_u_train, u_train]
