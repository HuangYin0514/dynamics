import numpy as np
import scipy.io
import torch.backends.cudnn

from trainer import Trainer
from utils import get_device, to_tensor, to_numpy

device = get_device()


# Geneate test data corresponding to one input sample
def generate_one_test_data(idx, usol, P):
    u = usol[idx]
    u0 = u[0, :]

    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, P)
    T, X = np.meshgrid(t, x)

    s = u.T.flatten()
    u = np.tile(u0, (P ** 2, 1))
    y = np.hstack([T.flatten()[:, None], X.flatten()[:, None]])

    return u, y, s


def compute_error(trainer, idx, usol):
    P = 101

    u_test, y_test, s_test = generate_one_test_data(idx, usol, P)
    u_test = u_test.reshape(P ** 2, -1)
    y_test = y_test.reshape(P ** 2, -1)
    s_test = s_test.reshape(P ** 2, -1)

    s_pred = trainer.predict_s(to_tensor(u_test), to_tensor(y_test))

    error = np.linalg.norm(s_test - to_numpy(s_pred)) / np.linalg.norm(s_test)
    return error


if __name__ == '__main__':
    # data
    path = 'data/Burger.mat'  # Please use the matlab script to generate data
    data = scipy.io.loadmat(path)
    usol = np.array(data['output'])
    k = 0
    N_test = 10
    idx = np.arange(k, k + N_test)
    print('Test list index is : {}'.format(idx))

    # model
    model = torch.load('result/model_final.pkl').to(device)
    # model = torch.load('result/model_final.pkl', map_location=torch.device('cpu')).to(device)

    # trainer
    trainer = Trainer()
    trainer.model = model

    # # compute_error
    # errors = list(map(compute_error, np.tile([trainer], (idx.shape[0],)), idx, np.tile(usol, (usol.shape[0], 1, 1, 1))))
    errors = []
    for i in idx:
        error = compute_error(trainer, i, usol)
        errors.append(error)

    errors = np.array(errors)
    print('error list of s is {}'.format(errors))

    mean_error = errors.mean()
    print('Mean relative L2 error of s: {:.2e}'.format(mean_error))

    print("done.")
