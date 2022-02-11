import numpy as np
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import solve_ivp
if __name__ == '__main__':
    features = 2000
    num = 10
    length_scale = 0.2
    x = np.linspace(0, 1, num=features)[:, None]
    A = gp.kernels.RBF(length_scale=length_scale)(x)
    L = np.linalg.cholesky(A + 1e-13 * np.eye(features))
    gps = (L @ np.random.randn(features, num)).transpose()
    gp =gps[0]

    u = interpolate.interp1d(np.linspace(0, 1, num=gp.shape[-1]), gp, kind='cubic', copy=False,
                             assume_sorted=True)

    x = np.sort(np.random.rand(1))
    def pde_func(t,y):
        return u(t)
    y = solve_ivp(pde_func, [0, 1], [0], 'RK45', t_eval = x).y[0]

    u_sensors = u(np.linspace(0, 1, num=100))

    res = np.hstack([np.tile(u_sensors, (1, 1)), x[:, None], y[:, None]])

    plt.figure()
    plt.plot(np.linspace(0, 1, num=gp.shape[-1]),u(np.linspace(0, 1, num=gp.shape[-1])))
    plt.plot(np.linspace(0, 1, num=100),u_sensors,'x')
    plt.show()
    print(x)
