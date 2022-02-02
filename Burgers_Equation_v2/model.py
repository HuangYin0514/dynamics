import torch
from torch import nn


class MlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        return self.Tanh(self.fc(x))


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


class PINN(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks

        self.encoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh()
        )

        self.mlp = self._make_layer(MlpBlock, num_blocks)

        self.dam = DAM(in_dim=20)

        self.decoder = nn.Linear(20, 1)

    @staticmethod
    def _make_layer(block, num_blocks):
        layers = []

        for _ in range(num_blocks):
            layers.append(block(input_dim=20, output_dim=20))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        x= self.dam(x)
        out = self.decoder(x)
        return out


if __name__ == '__main__':
    # # test MlpBlock
    # net = MlpBlock(2, 20)
    # t_input = torch.randn(32, 2)
    # output = net(t_input)
    # print(net)
    # print(output.shape)

    # test DAM
    # net = DAM(2)
    # t_input = torch.randn(6, 2)
    # output = net(t_input)
    # print(net)
    # print(output.shape)
    # print(t_input)
    # print(output)

    # test PINN
    net = PINN(8)
    t_input = torch.randn(32, 2)
    output = net(t_input)
    print(net)
    print(output.shape)
