import numpy as np
import torch
from scipy import interpolate
from scipy.integrate import solve_ivp
from sklearn import gaussian_process as gp
from torch.utils import data

from utils import get_device


class IntegralData():
    '''
        Data for learning the antiderivative operator.
    '''

    def __init__(self):
        super().__init__()
        self.s0 = [0]  # DeepONet_ode init data
        self.sensors = 100  # u0 length
        self.p = 1  # grad number
        self.length_scale = 0.2  # RBF parameters
        self.train_num = 1000  # train dataset number
        self.test_num = 1000  # test dataset number
        self.__init_data()

    def __init_data(self):
        features = 2000
        train = self.__gaussian_process(self.train_num, features)
        test = self.__gaussian_process(self.test_num, features)
        self.X_train, self.y_train = self.__generate(train)
        self.X_test, self.y_test = self.__generate(test)

    def __generate(self, gps):
        def generate(gp):
            u = interpolate.interp1d(np.linspace(0, 1, num=gp.shape[-1]), gp, kind='cubic', copy=False,
                                     assume_sorted=True)
            # y = solve_ivp(lambda t, y: u(t), [0, 1], self.s0, 'RK45', x).y[0]
            x = np.sort(np.random.rand(self.p))

            def pde_func(t, y):
                return u(t)

            y = solve_ivp(pde_func, [0, 1], self.s0, 'RK45', x).y[0]
            u_sensors = u(np.linspace(0, 1, num=self.sensors))
            return np.hstack([np.tile(u_sensors, (self.p, 1)), x[:, None], y[:, None]])

        res = np.vstack(list(map(generate, gps)))
        return (res[..., :-2], res[..., -2:-1]), res[..., -1:]

    def __gaussian_process(self, num, features):
        x = np.linspace(0, 1, num=features)[:, None]
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + 1e-13 * np.eye(features))
        return (L @ np.random.randn(features, num)).transpose()


# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s):
        'Initialization'

        self.u = u
        self.y = y
        self.s = s

        self.N = self.s.shape[0]  # all dataset number
        self.batch_size = self.s.shape[0]

        self.device = get_device()

    def __getitem__(self, index):
        'Generate one batch of data'
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __to_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return x

    def __data_generation(self):
        'Generates data containing batch_size samples'
        idx = np.random.choice(self.N, (self.batch_size,), replace=False)
        u = self.u[idx, :]
        y = self.y[idx, :]
        s = self.s[idx, :]

        # Construct batch
        inputs = (self.__to_tensor(u), self.__to_tensor(y))
        outputs = self.__to_tensor(s)

        return inputs, outputs


if __name__ == '__main__':
    data = IntegralData()
    X_train, y_train = data.X_train, data.y_train
    train_dataset = DataGenerator(u=X_train[0], y=X_train[1], s=y_train)

    train_data = iter(train_dataset)
    train_batch = next(train_data)
    train_batch2 = next(train_data)
    print("done.")
