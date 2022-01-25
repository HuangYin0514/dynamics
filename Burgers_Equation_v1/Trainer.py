
import random

import numpy as np
import scipy.io
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

from model import DNN

try:
    from pyDOE import lhs
except ImportError:
    import sys
    sys.path.append('/kaggle/input/pylib-pydoe/MySitePackages')
    from pyDOE import lhs

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# the physics-guided neural network
class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        self.layers = layers
        self.nu = nu

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # iterations
        self.Adam_nIter = 500
        self.Current_Iter = 0

        # loss function
        self.MSELoss = torch.nn.MSELoss()

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(
            self.dnn.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if not torch.cuda.is_available():
            print("using cpu for optim...")
            self.optimizer = torch.optim.LBFGS(
                self.dnn.parameters(),
                lr=1.0,
                max_iter=5,
                max_eval=5,
                history_size=50,
                tolerance_grad=1e-5,
                tolerance_change=1.0 * np.finfo(float).eps,
                line_search_fn="strong_wolfe",  # can be "strong_wolfe"
            )
            self.Adam_nIter = 3

        self.loss = []
        self.loss_u = []
        self.loss_f = []

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """The pytorch autograd version of calculating residual"""
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        f = u_t + u * u_x - self.nu * u_xx
        return f

    def compute_loss(self):
        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = self.MSELoss(self.u, u_pred)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f

        return loss, loss_u, loss_f

    def loss_func(self):

        loss, loss_u, loss_f = self.compute_loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.Current_Iter += 1

        if self.Current_Iter % 100 == 0:
            self.loss.append(loss.item())
            self.loss_u.append(loss_u.item())
            self.loss_f.append(loss_f.item())

            print(
                "Current_iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e"
                % (self.Current_Iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss

    def train(self):
        self.dnn.train()

        # adam优化
        for epoch in range(self.Adam_nIter):

            loss, loss_u, loss_f = self.compute_loss()

            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()

            if epoch % 100 == 0:
                print(
                    "Adam ----> It: {}, Loss: {}".format(
                        epoch,
                        loss.item(),
                    )
                )

        # LBFGS 优化  Backward and optimize
        self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f
