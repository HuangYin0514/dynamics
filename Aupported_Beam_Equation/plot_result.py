
from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from model import *



if __name__ == '__main__':
    data = scipy.io.loadmat("pred.mat")
    u_pred = np.real(data["u_pred"])
    
    
    t = np.linspace(0, 1, 100).flatten()[:, None]
    x = np.linspace(0, 1, 256).flatten()[:, None]

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    print(U_pred.shape)
    