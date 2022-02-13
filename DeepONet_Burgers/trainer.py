import torch
from torch import nn

from model import DeepONet
from utils import get_device

device = get_device()


# the physics-guided neural network
class Trainer():
    def __init__(self):
        # Network initialization and evaluation functions
        self.model = DeepONet().to(device)

        # loss function
        self.criterion = nn.MSELoss()

        # Use optimizers to set optimizer initialization and update functions
        self.optimizer_Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )

        # Logger
        self.loss_log = []

    # Define DeepONet architecture
    def operator_net(self, u, t, x):
        y = torch.cat([t[:, None], x[:, None]], dim=1)
        outputs = self.model(u, y)

        return outputs

    # Define ds/dx
    def s_x_net(self, u, t, x):
        s = self.operator_net(u, t, x)
        s_x = torch.autograd.grad(s, x, grad_outputs=torch.ones_like(s), retain_graph=True, create_graph=True)[0]
        return s_x

    # Define PDE residual
    def residual_net(self, u, t, x):
        s = self.operator_net(u, t, x)
        s_t = torch.autograd.grad(s, t, grad_outputs=torch.ones_like(s), retain_graph=True, create_graph=True)[0]
        s_x = torch.autograd.grad(s, x, grad_outputs=torch.ones_like(s), retain_graph=True, create_graph=True)[0]
        s_xx = torch.autograd.grad(s_x, x, grad_outputs=torch.ones_like(s_x), retain_graph=True, create_graph=True)[0]

        res = s_t + s * s_x - 0.01 * s_xx
        return res

    def loss_ics(self, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        pred = self.operator_net(u, y[:, 0], y[:, 1])

        # Compute loss
        ic_pred = pred - outputs.flatten()
        loss = self.criterion(ic_pred.flatten(), torch.zeros_like(ic_pred))
        return loss

    def loss_bcs(self, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s_bc1_pred = self.operator_net(u, y[:, 0], y[:, 1])
        s_bc2_pred = self.operator_net(u, y[:, 2], y[:, 3])

        s_x_bc1_pred = self.s_x_net(u, y[:, 0], y[:, 1])
        s_x_bc2_pred = self.s_x_net(u, y[:, 2], y[:, 3])

        # Compute loss
        s_bc_pred = s_bc1_pred - s_bc2_pred
        loss_s_bc = self.criterion(s_bc_pred.flatten(), torch.zeros_like(s_bc_pred))

        s_x_bc_pred = s_x_bc1_pred - s_x_bc2_pred
        loss_s_x_bc = self.criterion(s_x_bc_pred.flatten(), torch.zeros_like(s_x_bc_pred))

        return loss_s_bc + loss_s_x_bc

    def loss_res(self, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        pred = self.residual_net(u, y[:, 0], y[:, 1])

        # Compute loss
        res_pred = pred - outputs.flatten()
        loss = self.criterion(res_pred.flatten(), torch.zeros_like(res_pred))
        return loss


    # Define total loss
    def loss(self, ics_batch, bcs_batch, res_batch):
        loss_ics = self.loss_ics(ics_batch)
        loss_bcs = self.loss_bcs(bcs_batch)
        loss_res = self.loss_res( res_batch)
        loss = 20 * loss_ics + loss_bcs + loss_res
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

    def predict(self, U_star, Y_star):
        self.model.eval()
        pred = self.model(U_star, Y_star)
        return pred
