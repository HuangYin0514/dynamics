import numpy as np
import torch
from torch import nn

from model import PINN
from utils import get_device

device = get_device()


# the physics-guided neural network
class Trainer():
    def __init__(self):
        # Network initialization and evaluation functions
        self.model = PINN().to(device)

        # loss function
        self.criterion = nn.MSELoss()

        # Use optimizers to set optimizer initialization and update functions
        self.optimizer_Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )

        # Logger
        self.loss_log = []

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

    # Define DeepONet architecture
    def operator_net(self, x, t):
        y = torch.cat([x, t], dim=1)
        s = self.model(y)
        return s

    # Define PDE residual
    def residual_net(self, x, t):
        s = self.operator_net(x, t)

        s_t = torch.autograd.grad(
            s, t, grad_outputs=torch.ones_like(s), retain_graph=True, create_graph=True
        )[0]
        s_x = torch.autograd.grad(
            s, x, grad_outputs=torch.ones_like(s), retain_graph=True, create_graph=True
        )[0]
        s_xx = torch.autograd.grad(
            s_x,
            x,
            grad_outputs=torch.ones_like(s_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        res = s_t + s * s_x - 0.01 / np.pi * s_xx
        return res

    def loss_ics(self, batch):
        # Fetch data
        inputs, outputs = batch
        y = inputs

        # Compute forward pass
        pred = self.operator_net(y[:, 0:1], y[:, 1:2])

        # Compute loss
        loss = torch.mean((pred - outputs) ** 2)
        return loss

    def loss_bcs(self, batch):
        # Fetch data
        inputs, outputs = batch
        y = inputs

        # Compute forward pass
        pred = self.operator_net(y[:, 0:1], y[:, 1:2])

        # Compute loss
        loss = torch.mean((pred - outputs) ** 2)
        return loss

    def loss_res(self, batch):
        # Fetch data
        inputs, outputs = batch
        y = inputs

        # Compute forward pass
        pred = self.residual_net(y[:, 0:1], y[:, 1:2])

        # Compute loss
        loss = torch.mean((pred - outputs) ** 2)
        return loss

    # Define total loss
    def loss(self, ics_batch, bcs_batch, res_batch):
        loss_ics = self.loss_ics(ics_batch)
        loss_bcs = self.loss_bcs(bcs_batch)
        loss_res = self.loss_res(res_batch)
        loss = loss_ics + loss_bcs + loss_res
        return loss

    def train_step(self, ics_batch, bcs_batch, res_batch):
        self.optimizer_Adam.zero_grad()
        loss = self.loss(ics_batch, bcs_batch, res_batch)
        loss.backward()
        self.optimizer_Adam.step()
        return loss

    def train(self, ics_dataset, bcs_dataset, res_dataset, nIter=10000):
        self.model.train()

        ics_data = iter(ics_dataset)
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        # Main training loop
        for it in range(nIter):
            # Fetch data
            ics_batch = next(ics_data)
            bcs_batch = next(bcs_data)
            res_batch = next(res_data)

            loss = self.train_step(ics_batch, bcs_batch, res_batch)

            if it % 1000 == 0:
                # Store losses
                self.loss_log.append(loss.item())

                print("Adam\tepoch:{}\tloss:{:.5}".format(it, loss.item()))

        self.counter = 0
        ics_batch = next(ics_data)
        bcs_batch = next(bcs_data)
        res_batch = next(res_data)

        def closure():
            self.optimizer_LBFGS.zero_grad()
            loss = self.loss(ics_batch, bcs_batch, res_batch)
            loss.backward()
            if self.counter % 200 == 0:
                print("LBFGS\t epoch:{}\t loss:{:.5}".format(self.counter, loss.item()))
            self.counter = self.counter + 1
            return loss

        self.optimizer_LBFGS.step(closure)

    def predict_s(self, x_star):
        self.model.eval()
        pred = self.operator_net(x_star[:, 0:1], x_star[:, 1:2])
        return pred

    # def predict_res(self, U_star, Y_star):
    #     self.model.eval()
    #     pred = self.residual_net(U_star, Y_star[:, 0], Y_star[:, 1])
    #     return pred
