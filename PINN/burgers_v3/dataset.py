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

        data = scipy.io.loadmat("data/burgers_shock.mat")

        t = data["t"].flatten()[:, None]
        x = data["x"].flatten()[:, None]
        exact = np.real(data["usol"]).T
        self.__init_data(exact, x, t)

    def __init_data(self, exact, x, t):
        x_mesh, t_mesh = np.meshgrid(x, t)  # X(n_t,n_x) T(n_t,n_x)

        # ic constraints
        y_ic = np.hstack((x_mesh[0:1, :].T, t_mesh[0:1, :].T))  # 左
        s_ic = exact[0:1, :].T
        self.y_ics_train, self.s_ics_train = self.generate_ics_training_data( y_ic, s_ic)

        # bc constraints
        y_bc1 = np.hstack((x_mesh[:, 0:1], t_mesh[:, 0:1]))  # 下
        s_bc1 = exact[:, 0:1]
        y_bc2 = np.hstack((x_mesh[:, -1:], t_mesh[:, -1:]))  # 上
        s_bc2 = exact[:, -1:]
        y_bc = np.vstack([y_bc1, y_bc2])
        s_bc = np.vstack([s_bc1, s_bc2])
        self.y_bcs_train, self.s_bcs_train = self.generate_bcs_training_data( y_bc, s_bc)

        # res constraints
        y_mesh = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))  # (n_x*n_t, 2)
        lb = y_mesh.min(axis=0)  # [-1.0, 0.0]
        ub = y_mesh.max(axis=0)  # [1.0, 0.99]
        y_ibc = np.vstack([y_ic, y_bc1, y_bc2])
        self.y_res_train, self.s_res_train = self.generate_res_training_data( lb, ub, y_ibc)

    # ic constraints
    def generate_ics_training_data(self, y_ic, s_ic, n_ic=100):
        idx = np.random.choice(s_ic.shape[0], n_ic, replace=False)
        y = y_ic[idx, :]
        s = s_ic[idx, :]
        return  y, s

    # ic constraints
    def generate_bcs_training_data(self, y_bc, s_bc, n_bc=100):
        idx = np.random.choice(s_bc.shape[0], n_bc, replace=False)
        y = y_bc[idx, :]
        s = s_bc[idx, :]
        return  y, s

    # res constraints
    def generate_res_training_data(self, lb, ub, y_ibc, n_res=10000):
        y_sample = lb + (ub - lb) * lhs(2, n_res)
        y = np.vstack([y_sample, y_ibc])
        s = np.zeros((y.shape[0], 1))
        return  y, s


class DataGenerator(data.Dataset):
    def __init__(self, y, s):
        'Initialization'

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
        y = self.y[idx, :]
        s = self.s[idx, :]

        # Construct batch
        inputs =  to_tensor(y)
        outputs = to_tensor(s)

        return inputs, outputs


if __name__ == '__main__':
    burgerData = BurgerData()

    y_ics_train, s_ics_train = burgerData.y_ics_train, burgerData.s_ics_train
    y_bcs_train, s_bcs_train = burgerData.y_bcs_train, burgerData.s_bcs_train
    y_res_train, s_res_train = burgerData.y_res_train, burgerData.s_res_train

    ics_dataset = DataGenerator( y_ics_train, s_ics_train)
    bcs_dataset = DataGenerator( y_bcs_train, s_bcs_train)
    res_dataset = DataGenerator( y_res_train, s_res_train)

    output1 = next(iter(ics_dataset))
    output2 = next(iter(bcs_dataset))
    output3 = next(iter(res_dataset))
