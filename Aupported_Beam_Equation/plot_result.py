
from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from model import *



if __name__ == '__main__':
    data = scipy.io.loadmat("pred-7.mat")
    u_pred = np.real(data["u_pred"])
    
    
    t = np.linspace(0, 1, 100).flatten()[:, None]
    x = np.linspace(0, 1, 256).flatten()[:, None]

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    print(U_pred.shape)
    
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
    
    ################################
    fig = plt.figure()
    # fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    ax.plot(t, U_pred[:, 1]*1000, "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$x = 1$", fontsize=15)
    ax.axis("square")
    
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-3, 3])
    
    plt.show()