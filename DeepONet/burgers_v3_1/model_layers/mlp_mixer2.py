import torch
from torch import nn


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

class MlpMixer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh()
        )

        self.mlp = self._make_layer(MlpBlock, num_blocks=7)


        self.decoder = nn.Linear(40, 40)

    @staticmethod
    def _make_layer(block, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(input_dim=40, output_dim=40))

        return nn.Sequential(*layers)

    def forward(self, x):
        t = x[:, 0:1]
        x = x[:, 1:2]
        x = torch.cat([t, x], axis=1)
        x = self.encoder(x)
        x = self.mlp(x)
        out = self.decoder(x)
        return out

if __name__ == '__main__':
    mlp_mixer = MlpMixer()

    y = torch.randn(50, 2)
    output = mlp_mixer(y)
    print(output.shape)
