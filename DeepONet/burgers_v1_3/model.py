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


class PINN(nn.Module):
    def __init__(self, num_blocks=7):
        super().__init__()
        self.num_blocks = num_blocks

        self.encoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh()
        )

        self.mlp = self._make_layer(MlpBlock, num_blocks)

        self.dam = DAM(in_dim=20)

        self.decoder = nn.Linear(20, 40)

    @staticmethod
    def _make_layer(block, num_blocks):
        layers = []

        for _ in range(num_blocks):
            layers.append(block(input_dim=20, output_dim=20))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        x = self.dam(x)
        out = self.decoder(x)
        return out


class BranchNet(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.first_layers = MlpBlock(input_dim=256, output_dim=40)
        self.branch_layers = self._make_layer(MlpBlock, 1)

    @staticmethod
    def _make_layer(block, num_blocks):
        layers = []

        for _ in range(num_blocks):
            layers.append(block(input_dim=40, output_dim=40))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_layers(x)
        out = self.branch_layers(x)
        return out


class DeepONet(nn.Module):
    """
        Deep operator network.
        Input: ([batch size, branch_dim], [batch size, trunk_dim])
        Output: [batch size, 1]
    """

    def __init__(self):
        super().__init__()
        self.branch_net = BranchNet()
        self.trunk_net = PINN()
        self.net_bias = nn.Parameter(torch.zeros([1]))

        # self.branch_net.apply(weights_init_xavier_normal)
        # self.trunk_net.apply(weights_init_xavier_normal)

    def forward(self, u, y):
        B = self.branch_net(u)
        T = self.trunk_net(y)
        outputs = torch.sum(B * T, dim=1) + self.net_bias
        return outputs[:,None]


if __name__ == '__main__':
    model = DeepONet()
    print(model)
    u = torch.randn(20, 256)
    y = torch.randn(20, 2)
    outputs = model(u, y)
    print(outputs)
