from math import pi, sqrt

import numpy as np
from numpy import flipud


def GRF(N, m, gamma, tau, sigma, type):
    if type == "dirichlet":
        m = 0

    if type == "periodic":
        my_const = 2 * pi
    else:
        my_const = pi

    my_eigs = sqrt(2) * (abs(sigma) * ((my_const * np.arange(1, N + 1).T) ** 2 + tau ** 2) ** (-gamma / 2))

    if type == "dirichlet":
        alpha = np.zeros((N, 1))
    else:
        xi_alpha = np.random.randn(int(N), 1)
        alpha = my_eigs * xi_alpha

    if type == "neumann":
        beta = np.zeros((N, 1))
    else:
        xi_beta = np.random.randn(int(N), 1)
        beta = my_eigs * xi_beta

    a = alpha / 2
    b = -beta / 2

    c = np.vstack((flipud(a) - flipud(b) * 1j, np.ones_like(a)*m + 0 * 1j, a + b * 1j))
    print()
