import math

import matplotlib.pyplot as plt
import numpy as np


def runge_kutta(y, x, dx, f):
    """ y is the initial value for y
        x is the initial value for x
        dx is the time step in x
        f is derivative of function y(t)
    """
    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5 * k1, x + 0.5 * dx)
    k3 = dx * f(y + 0.5 * k2, x + 0.5 * dx)
    k4 = dx * f(y + k3, x + dx)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.


# compute y` = t * y^(1/2)
def func(y, t):
    return t * math.sqrt(y)


if __name__ == '__main__':
    t = 0.
    t_max = 10
    y = 1.
    dt = .1
    ys, ts = [], []

    # t = np.linspace(t_start,t_max, num=int(t_max/dt))

    while t <= t_max:
        y = runge_kutta(y, t, dt, func)
        t += dt
        ys.append(y)
        ts.append(t)

    exact = [(t ** 2 + 4) ** 2 / 16. for t in ts]

    plt.plot(ts, ys, label='runge_kutta')
    plt.plot(ts, exact, label='exact')
    plt.plot(ts, np.array(ys) - np.array(exact), label='error')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()
