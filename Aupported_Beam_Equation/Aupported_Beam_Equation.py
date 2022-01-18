import os

import numpy as np
import scipy.io
import torch

from datasets import InitialConditionDataset
from model import *

np.random.seed(1234)

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
    layers = [2, 20, 20, 20, 1]

    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 0.99])

    (
        X_star,
        X_f_train,
        X_u_train,
        u_train,
        X_u_derivative_xx_trian,
        u_derivative_xx_trian,
        X_u_derivative_t_trian,
        u_derivative_t_trian,
    ) = InitialConditionDataset(N_u, N_f, lb, ub).getData()

    model = PhysicsInformedNN(
        X_u_train,
        u_train,
        X_u_derivative_xx_trian,
        u_derivative_xx_trian,
        X_u_derivative_t_trian,
        u_derivative_t_trian,
        X_f_train,
        layers,
        lb,
        ub,
        nu,
    )
    model.train()

    u_pred, f_pred = model.predict(X_star)

    result_path = "result/" + "Aupported_Beam_Equation/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    scipy.io.savemat(result_path + "pred.mat", {"u_pred": u_pred})
