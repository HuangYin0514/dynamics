import numpy as np
import torch

from model import PINN

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# the physics-guided neural network
class PhysicsInformedNN:
    def __init__(self, training_loader):

        self.training_loader = training_loader

        self.nu = 0.01 / np.pi

        # deep neural networks
        self.model = PINN(num_blocks=8).to(device)

        # iterations
        self.epochs = 500

        # loss function
        self.MSELoss = torch.nn.MSELoss()

        # optimizers: using the same settings
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.loss = []
        self.loss_u = []
        self.loss_f = []

        if not torch.cuda.is_available():
            print("using cpu for optim...")
            self.optimizer_LBFGS = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=5,
                max_eval=5,
                history_size=50,
                tolerance_grad=1e-5,
                tolerance_change=1.0 * np.finfo(float).eps,
                line_search_fn="strong_wolfe",  # can be "strong_wolfe"
            )
            self.epochs = 1

    def net_u(self, x, t):
        u = self.model(torch.cat([x, t], dim=1))
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

    def train(self):
        self.model.train()

        # # Adam
        # for epoch in range(self.epochs):
        #     for iteration, data in enumerate(self.training_loader):
        #         x_f_train, x_u_train, u_train = data
        #         x_f_train, x_u_train, u_train = x_f_train[0], x_u_train[0], u_train[0]
        #         # data
        #         x_u = x_u_train[:, 0:1].clone().detach().requires_grad_(True).float().to(device)
        #         t_u = x_u_train[:, 1:2].clone().detach().requires_grad_(True).float().to(device)
        #         x_f = x_f_train[:, 0:1].clone().detach().requires_grad_(True).float().to(device)
        #         t_f = x_f_train[:, 1:2].clone().detach().requires_grad_(True).float().to(device)
        #         u = u_train.clone().detach().float().to(device)
        #
        #         def closure():
        #             self.optimizer_Adam.zero_grad()
        #             u_pred = self.net_u(x_u, t_u)
        #             f_pred = self.net_f(x_f, t_f)
        #             loss_u = self.MSELoss(u, u_pred)
        #             loss_f = torch.mean(f_pred ** 2)
        #             loss = loss_u + loss_f
        #             loss.backward()
        #             if epoch % 100 == 0:
        #                 print("Adam\tepoch:{}\tloss:{:.5}\tloss_u:{:.5}\tloss_f:{:.5}".format(epoch, loss.item(),
        #                                                                                       loss_u.item(),
        #                                                                                       loss_f.item()))
        #             return loss
        #
        #         self.optimizer_Adam.step(closure)

        # LBFGS
        for iteration, data in enumerate(self.training_loader):
            x_f_train, x_u_train, u_train = data
            x_f_train, x_u_train, u_train = x_f_train[0], x_u_train[0], u_train[0]
            # data
            x_u = x_u_train[:, 0:1].clone().detach().requires_grad_(True).float().to(device)
            t_u = x_u_train[:, 1:2].clone().detach().requires_grad_(True).float().to(device)
            x_f = x_f_train[:, 0:1].clone().detach().requires_grad_(True).float().to(device)
            t_f = x_f_train[:, 1:2].clone().detach().requires_grad_(True).float().to(device)
            u = u_train.clone().detach().float().to(device)

            self.epochs = 0

            def closure():
                self.optimizer_LBFGS.zero_grad()
                u_pred = self.net_u(x_u, t_u)
                f_pred = self.net_f(x_f, t_f)
                loss_u = torch.mean((u - u_pred) ** 2)
                loss_f = torch.mean(f_pred ** 2)
                loss = loss_u + loss_f
                loss.backward()
                self.loss.append(loss.item())
                self.loss_u.append(loss_u.item())
                self.loss_f.append(loss_f.item())
                if self.epochs % 100 == 0:
                    print("LBFGS\tepoch:{}\tloss:{:.5}\tloss_u:{:.5}\tloss_f:{:.5}".format(self.epochs, loss.item(),
                                                                                           loss_u.item(),
                                                                                           loss_f.item()))
                self.epochs = self.epochs + 1
                return loss

            self.optimizer_LBFGS.step(closure)
            break

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.model.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f
