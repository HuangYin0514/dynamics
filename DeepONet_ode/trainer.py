import torch
from torch import nn

from model import DeepONet

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# the physics-guided neural network
class Trainer():
    def __init__(self):
        # Network initialization and evaluation functions
        self.model = DeepONet()

        # loss function
        self.criterion = nn.MSELoss()

        # Use optimizers to set optimizer initialization and update functions
        self.optimizer_Adam = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Logger
        self.loss_log = []

    def loss_grad(self, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        pred = self.model(u, y)

        # Compute loss
        loss = self.criterion(pred.flatten(), outputs.flatten())
        return loss

    # Define total loss
    def loss(self, train_batch):
        loss_grad = self.loss_grad(train_batch)
        loss = loss_grad
        return loss

    def train_step(self, train_batch):
        self.optimizer_Adam.zero_grad()
        loss = self.loss(train_batch)
        self.optimizer_Adam.step()
        return loss

    def train(self, train_dataset, nIter=10000):
        self.model.train()

        train_data = iter(train_dataset)
        # Main training loop
        for it in range(nIter):
            # Fetch data
            train_batch = next(train_data)

            loss = self.train_step(train_batch)

            if it % 1 == 0:
                # Store losses
                self.loss_log.append(loss.item())

                print("Adam\tepoch:{}\tloss:{:.5}".format(it, loss.item()))

    def predict(self, U_star, Y_star):
        self.model.eval()
        pred = self.model(U_star, Y_star)
        return pred
