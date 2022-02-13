import numpy as np
import scipy.io
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from trainer import Trainer
from utils import get_device, to_numpy, to_tensor

device = get_device()


def plot(trainer):
    path = 'data/Burger.mat'  # Please use the matlab script to generate data
    data = scipy.io.loadmat(path)
    usol = np.array(data['output'])

    # Plot for one generated data
    k = 8  # index
    u = usol[k, :, :]
    u0 = usol[k, 0, :]

    P_test = 101

    t = np.linspace(0, 1, P_test)
    x = np.linspace(0, 1, P_test)
    T, X = np.meshgrid(t, x)

    u_test = np.tile(u0, (P_test ** 2, 1))
    y_test = np.hstack([T.flatten()[:, None], X.flatten()[:, None]])

    s_pred = trainer.predict_s(to_tensor(u_test), to_tensor(y_test))[:, None]
    s_pred  = to_numpy(s_pred)
    S_pred = griddata(y_test, s_pred.flatten(), (T, X), method='cubic')

    error_s = np.linalg.norm(u - S_pred.T, 2) / np.linalg.norm(u, 2)

    print("error_s: {:.3e}".format(error_s))

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(T, X, u, cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Exact $s(x,t)$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(T, X, S_pred.T, cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predict $s(x,t)$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(T, X, np.abs(S_pred.T - u), cmap='jet')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Absolute error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('result/deeponet.png')
    # plt.show()


if __name__ == '__main__':
    model = torch.load('result/model_final.pkl').to(device)
    trainer = Trainer()
    trainer.model = model
    plot(trainer)
