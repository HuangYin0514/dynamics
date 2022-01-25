import random
import sys

import numpy as np
import scipy.io
import torch
from scipy.interpolate import griddata

from Trainer import PhysicsInformedNN

try:
    from pyDOE import lhs
except ImportError:

    sys.path.append('/kaggle/input/pylib-pydoe/MySitePackages')
    from pyDOE import lhs


def init_random_state():
    random_seed = 1234
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    # speed up compution
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


init_random_state()

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == '__main__':
    nu = 0.01 / np.pi
    noise = 0.0

    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat("data/burgers_shock.mat")
    Exact = np.real(data["usol"]).T

    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1.0])  # (X,T)

    n_t = 100
    n_x = 256

    t = np.linspace(lb[1], ub[1], n_t).flatten()[:, None]
    x = np.linspace(lb[0], ub[0], n_x).flatten()[:, None]

    X, T = np.meshgrid(x, t)  # X(n_t,n_x) T(n_t,n_x)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # (n_x*n_t, 2)
    u_star = Exact.flatten()[:, None]

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))  # 左
    uu1 = -1 * np.sin(xx1 * np.pi)[:, 0][:, None]
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))  # 下
    uu2 = np.zeros([n_t, 1])
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))  # 上
    uu3 = np.zeros([n_t, 1])

    # all bounds constraints
    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])

    # pde constraints
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))

    # ib constraints
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
    model.train()

    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print("Error u: %e" % error_u)

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    Error = np.abs(Exact - U_pred)
