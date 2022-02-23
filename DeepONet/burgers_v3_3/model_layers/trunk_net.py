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


class MixMlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # x: (bs,tokens,channels) or (bs,channels,tokens)
        return self.fc2(self.gelu(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim=16, channels_mlp_dim=1024, tokens_hidden_dim=32, channels_hidden_dim=1024):
        super().__init__()
        self.ln = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp_block = MixMlpBlock(tokens_mlp_dim, mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block = MixMlpBlock(channels_mlp_dim, mlp_dim=channels_hidden_dim)

    def forward(self, x):
        """
        x: (bs,tokens,channels)
        """
        ### tokens mixing
        y = self.ln(x)
        y = y.transpose(1, 2)  # (bs,channels,tokens)
        y = self.tokens_mlp_block(y)  # (bs,channels,tokens)
        ### channels mixing
        y = y.transpose(1, 2)  # (bs,tokens,channels)
        out = x + y  # (bs,tokens,channels)
        y = self.ln(out)  # (bs,tokens,channels)
        y = out + self.channels_mlp_block(y)  # (bs,tokens,channels)
        return y


class mlp_blocks(nn.Module):
    def __init__(self):
        super().__init__()
        mlp_blocks = []
        for _ in range(5):
            mlp_blocks.append(
                MixerBlock(tokens_mlp_dim=2, channels_mlp_dim=40, tokens_hidden_dim=2, channels_hidden_dim=40))

        self.mlp_blocks = nn.Sequential(*mlp_blocks)

    def forward(self, x):
        x = self.mlp_blocks(x)
        return x


class trunk_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.t_fc = nn.Sequential(
            nn.Linear(1, 40),
            nn.Tanh()
        )

        self.x_fc = nn.Sequential(
            nn.Linear(1, 40),
            nn.Tanh()
        )

        self.encoder = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh()
        )

        self.mlp_blocks = mlp_blocks()

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
        t = y[:, 0:1]
        x = y[:, 1:2]
        t = self.t_fc(t).unsqueeze(axis=1)
        x = self.x_fc(x).unsqueeze(axis=1)
        x = torch.cat([t, x], 1)
        x = self.mlp_blocks(x)
        x = torch.mean(x, dim=1, keepdim=False)  # bs,channels
        # x = self.dam(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    y = torch.randn(30, 2)
    model = trunk_net()
    output = model(y)
    print(output.shape)
