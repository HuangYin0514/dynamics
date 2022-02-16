import numpy as np
import scipy.io
import torch.backends.cudnn

from trainer import Trainer
from utils import get_device, to_tensor, to_numpy

device = get_device()


# Geneate test data corresponding to one input sample
def generate_one_test_data():
    data = scipy.io.loadmat("data/burgers_shock.mat")

    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]

    exact = np.real(data["usol"]).T

    x_mesh, t_mesh = np.meshgrid(x, t)  # X(n_t,n_x) T(n_t,n_x)

    # Prediction
    x_star = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))  # (n_x*n_t, 2)
    s_star = exact.flatten()[:, None]
    u_star = np.tile(exact[0:1, :].T.flatten(), (s_star.shape[0], 1))

    return u_star, x_star, s_star


def compute_error(trainer):
    u_star,x_test, s_test = generate_one_test_data()

    s_pred = trainer.predict_s(to_tensor(u_star) , to_tensor(x_test))[:,None]

    error = np.linalg.norm(s_test - to_numpy(s_pred)) / np.linalg.norm(s_test)
    return error


if __name__ == '__main__':
    # model
    model = torch.load('result/model_final.pkl').to(device)

    # trainer
    trainer = Trainer()
    trainer.model = model

    # # compute_error
    errors = compute_error(trainer)
    errors = np.array(errors)
    mean_error = errors.mean()
    print('Mean relative L2 error of s: {:.2e}'.format(mean_error))

    print("done.")
