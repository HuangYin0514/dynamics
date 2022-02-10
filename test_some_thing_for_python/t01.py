from random import random

import numpy as np


# Geneate operator training data corresponding to one input sample
def generate_one_op_training_data(key, u, y, s, N_op=1000):
    x = y[:, 0]
    t = y[:, 1]
    Nx = len(x)
    Nt = len(t)
    idx = random.randint(key, (N_op, 2), 0, max(Nx, Nt))
    u_op = np.tile(u, (N_op, 1))
    y_op = np.hstack((x[idx[:, 0]][:, None], t[idx[:, 1]][:, None]))
    s_op = s[idx[:, 0], idx[:, 1]]
    return u_op, y_op, s_op

if __name__ == "__main__":
    x = np.linspace(0, 1, 256).flatten()[:, None]

    res = 0.04 * x * (1 - x)

    print(res)
    print()
