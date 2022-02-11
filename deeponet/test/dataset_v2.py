import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from sklearn import gaussian_process as gp
from torch.utils import data
import matplotlib.pyplot as plt

class IntegralData(data.Dataset):
    '''
        Data for learning the antiderivative operator.
    '''

    def __init__(self):
        super().__init__()
        self.s0 = [0]
        self.sensors = 100
        self.p = 1
        self.length_scale = 0.2
        self.train_num = 1000
        self.test_num = 1000
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

    data = IntegralData()
    X_train, y_train = data.X_train, data.y_train
    X_test, y_test = data.X_train, data.y_train

    print(data)

    plt.figure()
    plt.plot(np.linspace(0, 1, num=100),X_train[0][0], linestyle='--', color='blue', label='origin line')
    plt.plot(X_train[1][0], y_train[0], marker='x', color='red',label = 'grad')
    plt.legend()
    plt.show()

    print()
