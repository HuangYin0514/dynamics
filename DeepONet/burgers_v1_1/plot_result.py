import numpy as np
import scipy.io
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from trainer import Trainer
from utils import get_device, to_numpy, to_tensor

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
    u_star = np.tile(exact[0:1, :].T.flatten(), (x_star.shape[0], 1))

    return u_star, x_star, exact


def plot(trainer):
    u_star,x_test, exact = generate_one_test_data()

    t = np.linspace(-1, 1, 100).flatten()[:, None]
    x = np.linspace(-1, 1, 256).flatten()[:, None]

    s_pred = trainer.predict_s(to_tensor(u_star),to_tensor(x_test))[:, None]
    s_pred = to_numpy(s_pred)
    s_pred = s_pred.flatten()[:, None]
    s_pred = s_pred.reshape(100, 256)

    ######################################################################
    fig = plt.figure(figsize=(9, 25))

    ax = fig.add_subplot(611)
    h = ax.imshow(exact.T, interpolation='nearest', cmap='rainbow', extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    # 右侧条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.set_title('$u(t,x)_{exact}$', fontsize=20)  # font size doubled
    ax.tick_params(labelsize=15)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    # ------------------------------------------------------------

    ax = fig.add_subplot(612)
    h = ax.imshow(s_pred.T, interpolation='nearest', cmap='rainbow', extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    # 右侧条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.set_title('$u(t,x)_{pred}$', fontsize=20)  # font size doubled
    ax.tick_params(labelsize=15)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    # ------------------------------------------------------------

    ax = fig.add_subplot(613)
    h = ax.imshow(exact.T - s_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    # 右侧条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.set_title('$U_{errors}$', fontsize=20)  # font size doubled
    ax.tick_params(labelsize=15)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ######################################################################
    ax = fig.add_subplot(614)
    ax.plot(x, exact[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, s_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$u(t,x)$', size=20)
    ax.set_title('$t = 0.25$', fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = fig.add_subplot(615)
    ax.plot(x, exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, s_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$u(t,x)$', size=20)
    ax.set_title('$t = 0.25$', fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = fig.add_subplot(616)
    ax.plot(x, exact[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, s_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$', size=20)
    ax.set_ylabel('$u(t,x)$', size=20)
    ax.set_title('$t = 0.25$', fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ######################################################################
    plt.tight_layout()

    plt.savefig("result/deeponet.png")
    plt.show()


if __name__ == '__main__':
    model = torch.load('result/model_final.pkl').to(device)
    trainer = Trainer()
    trainer.model = model
    plot(trainer)
