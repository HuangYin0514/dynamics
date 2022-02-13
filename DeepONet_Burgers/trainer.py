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
        s_x = torch.autograd.grad(
            s, x, grad_outputs=torch.ones_like(s), retain_graph=True, create_graph=True
        )[0]
        return s_x

    def loss_ics(self, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        pred = self.operator_net(u, y[:, 0], y[:, 1])

        # Compute loss
        loss = self.criterion(pred.flatten(), outputs.flatten())
        return loss

    def loss_bcs(self, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        pred = self.model(u, y)

        # Compute loss
        loss = self.criterion(pred.flatten(), outputs.flatten())
        return loss

    # Define total loss
    def loss(self, ics_batch, bcs_batch, res_batch):
        loss_ics = self.loss_ics(ics_batch)
        # loss_bcs = self.loss_bcs(bcs_batch)
        # loss_res = self.loss_res( res_batch)
        # loss = 20 * loss_ics + loss_bcs + loss_res
        loss = loss_ics
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
