from re import L
import torch
from collections import OrderedDict
import numpy as np

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))

        layer_list.append(
            ("layer_%d" % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


# the physics-guided neural network
class PhysicsInformedNN:
    def __init__(
        self,
        X_u,
        u,
        X_u_derivative_xx,
        u_derivative_xx,
        X_u_derivative_t,
        u_derivative_t,
        X_f,
        layers,
        lb,
        ub,
        nu,
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
        
        self.x_u_derivative_xx = torch.tensor(X_u_derivative_xx[:, 0:1], requires_grad=True).float().to(device)
        self.t_u_derivative_xx = torch.tensor(X_u_derivative_xx[:, 1:2], requires_grad=True).float().to(device)
        self.u_derivative_xx = torch.tensor(u_derivative_xx).float().to(device)
        
        self.x_u_derivative_t = torch.tensor(X_u_derivative_t[:, 0:1], requires_grad=True).float().to(device)
        self.t_u_derivative_t = torch.tensor(X_u_derivative_t[:, 1:2], requires_grad=True).float().to(device)
        self.u_derivative_t = torch.tensor(u_derivative_t).float().to(device)
        
        self.layers = layers
        self.nu = nu

        Rho = 1000
        E = 1.5e+8
        I = (2e-6) / 3
        P = 1000
        v = -0.15
        self.pde_param1 = Rho
        self.pde_param2 = 2 * Rho * v
        self.pde_param3 = P - Rho * v ** 2
        self.pde_param4 = E * I

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1,
            max_iter=5000,
            max_eval=5000,
            history_size=50,
            tolerance_grad=1e-2,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        )

        # self.optimizer = torch.optim.LBFGS(
        #     self.dnn.parameters(),
        #     lr=1.0,
        #     max_iter=5,
        #     max_eval=5,
        #     history_size=50,
        #     tolerance_grad=1e-5,
        #     tolerance_change=1.0 * np.finfo(float).eps,
        #     line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        # )

        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_u_t(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))

        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]

        return u_t

    def net_u_xx(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))

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

        return u_xx

    def net_f(self, x, t):
        """The pytorch autograd version of calculating residual"""
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]

        u_tt = torch.autograd.grad(
            u_t,
            t,
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_xt = torch.autograd.grad(
            u_x,
            t,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_xx = torch.autograd.grad(
            u_x,
            x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_xxx = torch.autograd.grad(
            u_xx,
            x,
            grad_outputs=torch.ones_like(u_xx),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_xxxx = torch.autograd.grad(
            u_xxx,
            x,
            grad_outputs=torch.ones_like(u_xxx),
            retain_graph=True,
            create_graph=True,
        )[0]

        f = (
            self.pde_param1 * u_tt
            + self.pde_param2 * u_xt
            - self.pde_param3 * u_xx
            + self.pde_param4 * u_xxxx
        )
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)
        
        u_xx_pred = self.net_u_xx(self.x_u_derivative_xx, self.t_u_derivative_xx)  # TODO
        u_t_pred = self.net_u_t(self.x_u_derivative_t, self.t_u_derivative_t)  # TODO
        loss_u_xx = torch.mean((self.u_derivative_xx - u_xx_pred) ** 2)
        loss_u_t = torch.mean((self.u_derivative_t - u_t_pred) ** 2)

        loss = loss_u + loss_f + loss_u_xx + loss_u_t

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
