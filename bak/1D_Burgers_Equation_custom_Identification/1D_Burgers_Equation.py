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
    # Configurations
    nu = 0.01/np.pi

    N_u = 2000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('1D_Burgers_Equation_custom_Identification/data/burgers_shock.mat')

    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0) 


    # Training on Non-noisy Data
    noise = 0.0            

    # create training set
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]

    # training
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model.train(0)
    
    
    # evaluations
    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    lambda_1_value = model.lambda_1.detach().cpu().numpy()
    lambda_2_value = model.lambda_2.detach().cpu().numpy()
    lambda_2_value = np.exp(lambda_2_value)

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

    print('Error u: %e' % (error_u))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))  
    
    print('u_t + {}u*u_x - {}u_xx = 0'.format(lambda_1_value,lambda_2_value))
