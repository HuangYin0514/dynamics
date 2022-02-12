import numpy as np
import torch.backends.cudnn

from dataset import IntegralData
from trainer import Trainer
from utils import get_device, to_tensor, to_numpy

device = get_device()


# Compute relative l2 error over N test samples.
def compute_error(trainer, u, y, s):
    u_test, y_test, s_test = to_tensor(u), to_tensor(y), to_tensor(s)

    # Predict the solution and the residual
    s_pred = trainer.predict(u_test, y_test)[:, None]

    # Compute relative l2 error
    error_s = np.linalg.norm(to_numpy(s_test) - to_numpy(s_pred)) / np.linalg.norm(to_numpy(s_test))

    return error_s


if __name__ == '__main__':
    # data
    data = IntegralData()
    X_test, y_test = data.X_test, data.y_test

    # model
    model = torch.load('result/model_final.pkl').to(device)

    # trainer
    trainer = Trainer()
    trainer.model = model

    # compute_error
    error_s = compute_error(trainer, u=X_test[0], y=X_test[1], s=y_test)

    print('mean of relative L2 error of s: {:.2e}'.format(error_s.mean()))
    print('std of relative L2 error of s: {:.2e}'.format(error_s.std()))

