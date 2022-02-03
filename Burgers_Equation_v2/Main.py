import os
import random
import sys

import numpy as np
import scipy.io
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader

from Trainer import PhysicsInformedNN
from dataset import BurgersEquationDataSet

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
    torch.backends.cudnn.deterministic = True # speed up compution
    torch.backends.cudnn.benchmark = True


# init_random_state()
np.random.seed(1234)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == '__main__':
    train_set = BurgersEquationDataSet()
    training_loader = DataLoader(dataset=train_set,
                                 batch_size=len(train_set))
    model = PhysicsInformedNN(training_loader)
    model.train()

    # prediction
    x_star = train_set.x_star
    u_star = train_set.u_star
    x_u_train = train_set.x_u_train

    u_pred, f_pred = model.predict(x_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print("Error u: %e" % error_u)

    result_path = "result/Burgers_Equation/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    scipy.io.savemat(result_path + "pred.mat",
                     {"u_pred": u_pred, "x_u_train": x_u_train, "loss": model.loss, "loss_u": model.loss_u,
                      "loss_f": model.loss_f})
