import torch

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
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
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat("1D_Burgers_Equation_custom/data/burgers_shock.mat")

    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = np.real(data["usol"]).T

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    # 左 上 下 边界
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
    model.train()

    u_pred, f_pred = model.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print("Relatively Error u: %e" % (error_u))
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    Error = np.abs(Exact - U_pred)
    print("Absolute Error u: {}" .format(np.sum(Error)))
    


    ####### Row 0: u(t,x) ##################
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    h = ax.imshow(
        U_pred.T,
        interpolation="nearest",
        cmap="rainbow",
        extent=[t.min(), t.max(), x.min(), x.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data ({} points)".format(u_train.shape[0]),
        markersize=4,  # marker size doubled
        clip_on=False,
        alpha=1.0,
    )
    
    ax.plot(
        X_f_train[:, 1],
        X_f_train[:, 0],
        "kx",
        label="Data ({} points)".format(X_f_train.shape[0]),
        markersize=1,  # marker size doubled
        clip_on=False,
        alpha=1.0,
    )

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, "w-", linewidth=1)

    ax.set_xlabel("$t$", size=20)
    ax.set_ylabel("$x$", size=20)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={"size": 15},
    )
    ax.set_title("$u(t,x)$", fontsize=20)  # font size doubled
    ax.tick_params(labelsize=15)
    plt.savefig("pred.png")


    ####### Row 1: u(t,x) slices ##################
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(
        top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5
    )

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[25, :], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[25, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = 0.25$", fontsize=15)
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[50, :], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[50, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title("$t = 0.50$", fontsize=15)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        prop={"size": 15},
    )

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[75, :], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[75, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title("$t = 0.75$", fontsize=15)

    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(15)
    plt.savefig("item.png")


    ####### ERROR for model ##################
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    h = ax.imshow(
        np.abs((U_pred- Exact).T),
        interpolation="nearest",
        cmap="rainbow",
        extent=[t.min(), t.max(), x.min(), x.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel("$t$", size=20)
    ax.set_ylabel("$x$", size=20)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={"size": 15},
    )
    ax.set_title("$u(t,x)$", fontsize=20)  # font size doubled
    ax.tick_params(labelsize=15)

    plt.savefig("error.png")