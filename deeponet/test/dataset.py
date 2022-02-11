from functools import wraps

import numpy as np
import torch
from scipy import interpolate
from scipy.integrate import solve_ivp
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt

def map_elementwise(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        container, idx = None, None
        for arg in args:
            if type(arg) in (list, tuple, dict):
                container, idx = type(arg), arg.keys() if type(arg) == dict else len(arg)
                break
        if container is None:
            for value in kwargs.values():
                if type(value) in (list, tuple, dict):
                    container, idx = type(value), value.keys() if type(value) == dict else len(value)
                    break
        if container is None:
            return func(*args, **kwargs)
        elif container in (list, tuple):
            get = lambda element, i: element[i] if type(element) is container else element
            return container(wrapper(*[get(arg, i) for arg in args],
                                     **{key: get(value, i) for key, value in kwargs.items()})
                             for i in range(idx))
        elif container is dict:
            get = lambda element, key: element[key] if type(element) is dict else element
            return {key: wrapper(*[get(arg, key) for arg in args],
                                 **{key_: get(value_, key) for key_, value_ in kwargs.items()})
                    for key in idx}

    return wrapper


class Data:
    '''Standard data format.
    '''

    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.__device = None
        self.__dtype = None

    def get_batch(self, batch_size):
        @map_elementwise
        def batch_mask(X, num):
            return np.random.choice(X.size(0), num, replace=False)

        @map_elementwise
        def batch(X, mask):
            return X[mask]

        mask = batch_mask(self.y_train, batch_size)
        return batch(self.X_train, mask), batch(self.y_train, mask)

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.__to_cpu()
            self.__device = torch.device('cpu')
        elif d == 'gpu':
            self.__to_gpu()
            self.__device = torch.device('cuda')
        else:
            raise ValueError

    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.__to_float()
            self.__dtype = torch.float32
        elif d == 'double':
            self.__to_double()
            self.__dtype = torch.float64
        else:
            raise ValueError

    def __to_cpu(self):
        @map_elementwise
        def trans(d):
            if isinstance(d, np.ndarray):
                return torch.DoubleTensor(d)
            elif isinstance(d, torch.Tensor):
                return d.cpu()

        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))

    def __to_float(self):
        if self.device is None:
            raise RuntimeError('device is not set')

        @map_elementwise
        def trans(d):
            if isinstance(d, torch.Tensor):
                return d.float()

        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))

    def __to_double(self):
        if self.device is None:
            raise RuntimeError('device is not set')

        @map_elementwise
        def trans(d):
            if isinstance(d, torch.Tensor):
                return d.double()

        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            setattr(self, d, trans(getattr(self, d)))


class IntegralData(Data):
    '''Data for learning the antiderivative operator.
    '''

    def __init__(self, s0, sensors, p, length_scale, train_num, test_num):
        super().__init__()
        self.s0 = s0
        self.sensors = sensors
        self.p = p
        self.length_scale = length_scale
        self.train_num = train_num
        self.test_num = test_num
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
            x = np.sort(np.random.rand(self.p))
            y = solve_ivp(lambda t, y: u(t), [0, 1], self.s0, 'RK45', x).y[0]
            u_sensors = u(np.linspace(0, 1, num=self.sensors))
            return np.hstack([np.tile(u_sensors, (self.p, 1)), x[:, None], y[:, None]])

        res = np.vstack(list(map(generate, gps)))
        return (res[..., :-2], res[..., -2:-1]), res[..., -1:]

    def __gaussian_process(self, num, features):
        x = np.linspace(0, 1, num=features)[:, None]
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + 1e-13 * np.eye(features))
        return (L @ np.random.randn(features, num)).transpose()


if __name__ == '__main__':
    s0 = [0]
    sensors = 100
    p = 1
    length_scale = 0.2
    train_num = 1000
    test_num = 1000

    data = IntegralData(s0, sensors, p, length_scale, train_num, test_num)
    X_train, y_train = data.X_train, data.y_train


    print(data)
    # plt.show(data)
    print()
