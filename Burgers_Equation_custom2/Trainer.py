from re import L
import torch

import numpy as np
from model import DNN

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# the physics-guided neural network
class PhysicsInformedNN:
    def __init__(
        self,
        X_u,
        u,
        X_f,
        layers,
        lb,
        ub,
    ):

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

        # deep neural networks
        self.dnn = DNN(layers).to(device)

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

        self.param = 0.01 / np.pi
        self.iter = 0

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

        f = u_t + u * u_x - self.param * u_xx
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f

        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                "Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e"
                % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )

        return loss

    def train(self):
        self.dnn.train()

        # Backward and optimize
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
