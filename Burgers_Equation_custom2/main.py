import os
import random

import numpy as np
import scipy.io
import torch

from datasets import InitialConditionDataset
from Trainer import *


def init_random_state():
    random_seed = 1234
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.


init_random_state()


def speedup_for_torch():
    torch.backends.cudnn.deterministic = True
    # speed up compution
    torch.backends.cudnn.benchmark = True


speedup_for_torch()

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    nu = 0.01 / np.pi
    noise = 0.0

    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 1.0])  # (X,T)

    (
        X_star,
        X_f_train,
        X_u_train,
        u_train,
    ) = InitialConditionDataset(N_u, N_f, lb, ub).getData()

    model = PhysicsInformedNN(
        X_u_train,
        u_train,
        X_f_train,
        layers,
        lb,
        ub,
    )
    model.train()

    u_pred, f_pred = model.predict(X_star)

    result_path = "result/Burgers_Equation_custom2/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    scipy.io.savemat(result_path + "pred.mat", {"u_pred": u_pred})
