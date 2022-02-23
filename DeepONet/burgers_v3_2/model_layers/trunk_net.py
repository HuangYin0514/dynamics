import torch
import torch.nn as nn


class DAM(nn.Module):
    """ Discriminative Amplitude Modulator Layer (1-D) """

    def __init__(self, in_dim):
        super(DAM, self).__init__()
        self.in_dim = in_dim

        mu = torch.arange(self.in_dim).float() / self.in_dim * 5.0
        self.mu = nn.Parameter(mu, requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)
        self.register_parameter('mu', self.mu)
        self.register_parameter('beta', self.beta)
        self.register_parameter('alpha', self.alpha)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        return x * self.mask()

    def mask(self):
        return self.relu(self.tanh((self.alpha ** 2) * (self.mu + self.beta)))


class MlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mlpBlock_layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.mlpBlock_layers(x)
        return out


class trunk_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh()
        )

        self.mlp = self._make_layer(MlpBlock, num_blocks=7)

        self.dam = DAM(in_dim=40)

        self.decoder = nn.Linear(40, 40)

    @staticmethod
    def _make_layer(block, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(input_dim=40, output_dim=40))

        return nn.Sequential(*layers)

    def forward(self, y):
        # t = y[:, 0:1]
        # x = y[:, 1:2]
        # x = torch.cat([t, x], 1)
        x = self.encoder(y)
        x = self.mlp(x)
        x = self.dam(x)
        out = self.decoder(x)
        return out
