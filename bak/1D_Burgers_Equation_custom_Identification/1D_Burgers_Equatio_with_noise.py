import torch

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from model import *
import random

# env setting ==============================================================================
# Fix random seed
random_seed = 2021
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)  # Numpy module.
random.seed(random_seed)  # Python random module.
torch.backends.cudnn.deterministic = True
# speed up compution
torch.backends.cudnn.benchmark = True
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

    noise = 0.01    

    # create training set
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])

    # training
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model.train(10000)   
    
    
    # evaluations
    u_pred, f_pred = model.predict(X_star)

    lambda_1_value_noisy = model.lambda_1.detach().cpu().numpy()
    lambda_2_value_noisy = model.lambda_2.detach().cpu().numpy()
    lambda_2_value_noisy = np.exp(lambda_2_value_noisy)

    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu) / nu * 100

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    
    print('Error u: %e' % (error_u))    
    print('Error l1: %.5f%%' % (error_lambda_1_noisy))                             
    print('Error l2: %.5f%%' % (error_lambda_2_noisy))  
    
    print('u_t + {}u*u_x - {}u_xx = 0'.format(lambda_1_value_noisy,lambda_2_value_noisy))
