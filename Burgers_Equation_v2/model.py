import torch
from torch import nn


class MlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        return self.Tanh(self.fc(x))


class PINN(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.num_blocks = num_blocks


        self.encoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh()
        )

        self.mlp = self._make_layer(MlpBlock, num_blocks)

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
        out = self.decoder(x)
        return out


if __name__ == '__main__':
    # # test MlpBlock
    # net = MlpBlock(2, 20)
    # t_input = torch.randn(32, 2)
    # output = net(t_input)
    # print(net)
    # print(output.shape)

    net = PINN(8)
    t_input = torch.randn(32, 2)
    output = net(t_input)
    print(net)
    print(output.shape)
