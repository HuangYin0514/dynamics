import random

import numpy as np


# Geneate ics training data corresponding to one input sample
def generate_one_ics_training_data(key, u0, m=101, P=101):
    t_0 = np.zeros((P, 1))
    x_0 = np.linspace(0, 1, P)[:, None]

    y = np.hstack([t_0, x_0])
    u = np.tile(u0, (P, 1))
    s = u0

    return u, y, s


# Geneate bcs training data corresponding to one input sample
def generate_one_bcs_training_data(key, u0, m=101, P=100):
    t_bc = random.uniform(key, (P, 1))
    x_bc1 = np.zeros((P, 1))
    x_bc2 = np.ones((P, 1))

    y1 = np.hstack([t_bc, x_bc1])  # shape = (P, 2)
    y2 = np.hstack([t_bc, x_bc2])  # shape = (P, 2)

    u = np.tile(u0, (P, 1))
    y = np.hstack([y1, y2])  # shape = (P, 4)
    s = np.zeros((P, 1))

    return u, y, s


# Geneate res training data corresponding to one input sample
def generate_one_res_training_data(key, u0, m=101, P=1000):
    subkeys = random.split(key, 2)

    t_res = random.uniform(subkeys[0], (P, 1))
    x_res = random.uniform(subkeys[1], (P, 1))

    u = np.tile(u0, (P, 1))
    y = np.hstack([t_res, x_res])
    s = np.zeros((P, 1))

    return u, y, s


# Geneate test data corresponding to one input sample
def generate_one_test_data(idx, usol, m=101, P=101):
    u = usol[idx]
    u0 = u[0, :]

    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, P)
    T, X = np.meshgrid(t, x)

    s = u.T.flatten()
    u = np.tile(u0, (P ** 2, 1))
    y = np.hstack([T.flatten()[:, None], X.flatten()[:, None]])

    return u, y, s




if __name__ == "__main__":
    a = 0.0009685528930276632
    print("a:{:.5}".format(a))
