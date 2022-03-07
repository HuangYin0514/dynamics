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
            weight_decay=0.001
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_Adam, milestones=[150, 250], gamma=0.1)
        # Logger
        self.loss_log = []

    # Define DeepONet architecture
    def operator_net(self, q, q_t):
        x = torch.cat([q, q_t], 1)
        y = self.model(x)
        return y

    def loss_doublePendulum(self, doublePendulum_batch):
        # Fetch data
        inputs, outputs = doublePendulum_batch
        q, q_t = inputs[:, 0:2], inputs[:, 2:4]

        # Compute forward pass
        pred = self.operator_net(q, q_t)

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

    def train(self, train_loader, nIter=10000):
        self.model.train()

        # Main training loop
        for it in range(nIter):
            self.scheduler.step()

            # Fetch data
            for data in train_loader:
                loss = self.train_step(data)

            if it % 10 == 0:
                # Store losses
                self.loss_log.append(loss.item())

                print(
                    "Adam \t epoch:{} \t loss:{:.5}".format(it, loss.item()))

    def predict(self, q, q_t):
        self.model.eval()
        q_tt = self.operator_net(q, q_t)
        return q_tt
