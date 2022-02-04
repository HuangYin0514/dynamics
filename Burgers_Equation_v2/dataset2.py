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

        # 上下界
        self.lb = np.array([-1.0, 0.0])
        self.ub = np.array([1.0, 0.99])  # (X,T)

        # 时间和空间点数
        n_t = 100
        n_x = 256

        t = np.linspace(self.lb[1], self.ub[1], n_t).flatten()[:, None]
        x = np.linspace(self.lb[0], self.ub[0], n_x).flatten()[:, None]

        x_mesh, t_mesh = np.meshgrid(x, t)  # X(n_t,n_x) T(n_t,n_x)

        xx1 = np.hstack((x_mesh[0:1, :].T, t_mesh[0:1, :].T))  # 左
        uu1 = -1 * np.sin(xx1 * np.pi)[:, 0][:, None]
        xx2 = np.hstack((x_mesh[:, 0:1], t_mesh[:, 0:1]))  # 下
        uu2 = np.zeros([n_t, 1])
        xx3 = np.hstack((x_mesh[:, -1:], t_mesh[:, -1:]))  # 上
        uu3 = np.zeros([n_t, 1])

        # all bounds constraints
        x_u_train = np.vstack([xx1, xx2, xx3])
        u_train = np.vstack([uu1, uu2, uu3])

        # pde constraints
        x_f_train = self.lb + (self.ub - self.lb) * lhs(2, n_f)
        self.x_f_train = np.vstack((x_f_train, x_u_train))

        # ib constraints
        idx = np.random.choice(x_u_train.shape[0], n_u, replace=False)
        self.x_u_train = x_u_train[idx, :]
        self.u_train = u_train[idx, :]

        # Prediction
        data = scipy.io.loadmat("data/burgers_shock.mat")
        exact = np.real(data["usol"]).T
        self.x_star = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))  # (n_x*n_t, 2)
        self.u_star = exact.flatten()[:, None]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x_f_train = self.x_f_train
        x_u_train = self.x_u_train
        u_train = self.u_train

        return [x_f_train, x_u_train, u_train]
