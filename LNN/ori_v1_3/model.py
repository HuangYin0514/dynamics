import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.act = nn.Softplus()

    def forward(self, x):
        # x: (bs,tokens,channels) or (bs,channels,tokens)
        return self.act(self.fc1(x))


class L_mlp(nn.Module):
    """
        Deep operator network.
        Input: ([batch size, 4])
        Output: [batch size, 1]
    """

    def __init__(self):
        super().__init__()
        input_dim = 4
        output_dim = 1

        layers = []
        input_layer = nn.Linear(input_dim, 600)
        layers.append(input_layer)
        for _ in range(2):
            layers.append(MlpBlock(600, 600))
        output_layer = nn.Linear(600, output_dim)
        layers.append(output_layer)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out


if __name__ == '__main__':
    model = L_mlp()
    print(model)
    # x = torch.randn(20, 4)
    q = torch.randn(20, 2)
    qt = torch.randn(20, 2)
    x = torch.cat([q, qt], 1)
    outputs = model(x)
    print(outputs)
