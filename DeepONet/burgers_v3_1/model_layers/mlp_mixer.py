import torch
from torch import nn


class MlpBlock(nn.Module):
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
        self.tokens_mlp_block = MlpBlock(tokens_mlp_dim, mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block = MlpBlock(channels_mlp_dim, mlp_dim=channels_hidden_dim)

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


class T_MixerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.T_MixerBlock_layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
        )

    def forward(self, x):
        return self.T_MixerBlock_layers(x)


class X_MixerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.X_MixerBlock_layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
        )

    def forward(self, x):
        return self.X_MixerBlock_layers(x)


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, tokens_hidden_dim, channels_hidden_dim, tokens_mlp_dim,
                 channels_mlp_dim):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks  # num of mlp layers
        self.patch_size = patch_size
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.embd = nn.Conv2d(3, channels_mlp_dim, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(channels_mlp_dim)
        self.mlp_blocks = []
        for _ in range(num_blocks):
            self.mlp_blocks.append(MixerBlock(tokens_mlp_dim, channels_mlp_dim, tokens_hidden_dim, channels_hidden_dim))
        self.fc = nn.Linear(channels_mlp_dim, num_classes)

        # mine
        self.T_MixerBlock = T_MixerBlock()
        self.X_MixerBlock = X_MixerBlock()

    def forward(self, y):

        t = y[:, 0:1]
        x = y[:, 1:2]
        t = self.T_MixerBlock(t).unsqueeze(axis=1)
        x = self.X_MixerBlock(x).unsqueeze(axis=1)
        y = torch.cat([t, x], axis=1)

        print(y.device)

        for i in range(self.num_blocks):
            y = self.mlp_blocks[i](y)  # bs,tokens,channels
            print(y.device)

        y = self.ln(y)  # bs,tokens,channels
        y = torch.mean(y, dim=1, keepdim=False)  # bs,channels
        probs = self.fc(y)  # bs,num_classes

        return probs


if __name__ == '__main__':
    mlp_mixer = MlpMixer(num_classes=100, num_blocks=10, patch_size=10, tokens_hidden_dim=32, channels_hidden_dim=128,
                         tokens_mlp_dim=2, channels_mlp_dim=128)

    y = torch.randn(50, 2)
    output = mlp_mixer(y)
    print(output.shape)
