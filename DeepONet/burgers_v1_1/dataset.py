import numpy as np
import scipy.io
from torch.utils import data

from utils import get_device, to_tensor

try:
    from pyDOE import lhs
except ImportError:
    import sys

    sys.path.append('/kaggle/input/pylib-pydoe/MySitePackages')
    from pyDOE import lhs


class BurgerData():
    '''
        Data for learning the antiderivative operator.
    '''

    def __init__(self):
        super().__init__()

        n_u = 100
        n_f = 10000

        data = scipy.io.loadmat("data/burgers_shock.mat")

        t = data["t"].flatten()[:, None]
        x = data["x"].flatten()[:, None]
        exact = np.real(data["usol"]).T

        x_mesh, t_mesh = np.meshgrid(x, t)  # X(n_t,n_x) T(n_t,n_x)

        # Prediction
        self.x_star = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))  # (n_x*n_t, 2)
        self.u_star = exact.flatten()[:, None]

        # 上下界
        # lb = np.array([-1.0, 0.0])
        # ub = np.array([1.0, 0.99])  # (X,T)
        lb = self.x_star.min(axis=0)  # [-1.0, 0.0]
        ub = self.x_star.max(axis=0)  # [1.0, 0.99]

        xx1 = np.hstack((x_mesh[0:1, :].T, t_mesh[0:1, :].T))  # 左
        uu1 = exact[0:1, :].T
        xx2 = np.hstack((x_mesh[:, 0:1], t_mesh[:, 0:1]))  # 下
        uu2 = exact[:, 0:1]
        xx3 = np.hstack((x_mesh[:, -1:], t_mesh[:, -1:]))  # 上
        uu3 = exact[:, -1:]

        # all bounds constraints
        x_u_train = np.vstack([xx1, xx2, xx3])
        u_train = np.vstack([uu1, uu2, uu3])

        # pde constraints
        x_f_train = lb + (ub - lb) * lhs(2, n_f)
        self.x_res_train = np.vstack((x_f_train, x_u_train))
        self.s_res_train = np.zeros((self.x_res_train.shape[0], 1))
        self.u_res_train = np.tile(uu1.flatten(), (self.x_res_train.shape[0], 1))

        # ib constraints
        idx = np.random.choice(x_u_train.shape[0], n_u, replace=False)
        self.x_ibcs_train = x_u_train[idx, :]
        self.s_ibcs_train = u_train[idx, :]
        self.u_ibcs_train = np.tile(uu1.flatten(), (n_u, 1))


class DataGenerator(data.Dataset):
    def __init__(self, u, y, s):
        'Initialization'

        self.u = u
        self.y = y
        self.s = s

        self.N = self.s.shape[0]  # all dataset number
        self.batch_size = self.N

        self.device = get_device()

    def __getitem__(self, index):
        'Generate one batch of data'
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __data_generation(self):
        'Generates data containing batch_size samples'
        idx = np.random.choice(self.N, (self.batch_size,), replace=False)  # False: 同一个元素只能被选取一次。
        u = self.u[idx, :]
        y = self.y[idx, :]
        s = self.s[idx, :]

        # Construct batch
        inputs = (to_tensor(u), to_tensor(y))
        outputs = to_tensor(s)

        return inputs, outputs


if __name__ == '__main__':
    burgerData = BurgerData()

    x_ibcs_train, u_ibcs_train, s_ibcs_train = burgerData.x_ibcs_train, burgerData.u_ibcs_train, burgerData.s_ibcs_train
    x_res_train, u_res_train, s_res_train = burgerData.x_res_train, burgerData.u_res_train, burgerData.s_res_train

    ics_dataset = DataGenerator(x_ibcs_train, u_ibcs_train, s_ibcs_train)
    res_dataset = DataGenerator(x_res_train, u_res_train, s_res_train)

    outputs = next(iter(ics_dataset))
