import torch

from model import L_mlp
from utils import get_device

device = get_device()


# the physics-guided neural network
class Trainer():
    def __init__(self):
        # Network initialization and evaluation functions
        self.model = L_mlp().to(device)

        # Use optimizers to set optimizer initialization and update functions
        self.optimizer_Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
        )

        # Logger
        self.loss_log = []

    # Define DeepONet architecture
    def operator_net(self, q, q_t):
        x = torch.cat([q, q_t], 1)
        y = self.model(x)
        return y

    def get_qtt(self, L, q, q_t):

        L_qt = torch.autograd.grad(
            L, q_t, grad_outputs=torch.ones_like(L), retain_graph=True, create_graph=True
        )[0]

        L_qt_qt = torch.autograd.grad(
            L_qt, q_t, grad_outputs=torch.ones_like(L_qt), retain_graph=True, create_graph=True
        )[0]

        L_q = torch.autograd.grad(
            L, q, grad_outputs=torch.ones_like(L), retain_graph=True, create_graph=True
        )[0]

        L_q_qt = torch.autograd.grad(
            L_q, q_t, grad_outputs=torch.ones_like(L_q), retain_graph=True, create_graph=True
        )[0]

        q_tt = (1 / L_qt_qt) * (L_q - L_q_qt)

        return q_tt

    def loss_doublePendulum(self, doublePendulum_batch):
        # Fetch data
        inputs, outputs = doublePendulum_batch
        q, q_t = inputs[:, 0:2], inputs[:, 2:4]
        # Compute forward pass
        L = self.operator_net(q, q_t)

        q_tt = self.get_qtt(L, q, q_t)

        pred = torch.cat([q_t, q_tt], 1)

        # Compute loss
        loss = torch.mean((pred - outputs) ** 2)
        return loss

    # Define total loss
    def loss(self, doublePendulum_batch):
        loss = self.loss_doublePendulum(doublePendulum_batch)
        return loss

    def train_step(self, batch):
        self.optimizer_Adam.zero_grad()
        loss = self.loss(batch)
        loss.backward()
        self.optimizer_Adam.step()
        return loss

    def train(self, doublePendulum_dataset, nIter=10000):
        self.model.train()

        doublePendulum_data = iter(doublePendulum_dataset)

        # Main training loop
        for it in range(nIter):
            # Fetch data
            doublePendulum_batch = next(doublePendulum_data)

            loss = self.train_step(doublePendulum_batch)

            if it % 10 == 0:
                # Store losses
                self.loss_log.append(loss.item())

                print(
                    "Adam \t epoch:{} \t loss:{:.5}".format(it, loss.item()))

    def predict(self, y0):
        self.model.eval()

        q, q_t = y0[:, 0:2], y0[:, 2:4]
        L = self.operator_net(q, q_t)
        q_tt = self.get_qtt(L, q, q_t)

        return q_t, q_tt

    # def predict_res(self, U_star, Y_star):
    #     self.model.eval()
    #     pred = self.residual_net(U_star, Y_star[:, 0], Y_star[:, 1])
    #     return pred
