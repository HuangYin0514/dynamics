import torch
import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class MlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mlpBlock_layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.mlpBlock_layers(x)
        return out


class BranchNet(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.first_layers = MlpBlock(input_dim=100, output_dim=40)
        self.branch_layers = self._make_layer(MlpBlock, 3)

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


class TrunkNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_layers = MlpBlock(input_dim=1, output_dim=40)
        self.trunk_layers = self._make_layer(MlpBlock, 3)

    @staticmethod
    def _make_layer(block, num_blocks):
        layers = []

        for _ in range(num_blocks):
            layers.append(block(input_dim=40, output_dim=40))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_layers(x)
        out = self.trunk_layers(x)
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
        self.trunk_net = TrunkNet()
        self.net_bias = nn.Parameter(torch.zeros([1]))
        # self.branch_net.apply(weights_init_kaiming)

    def forward(self, u, y):
        B = self.branch_net(u)
        T = self.trunk_net(y)
        outputs = torch.sum(B * T, dim=1)+self.net_bias
        return outputs


if __name__ == '__main__':
    model = DeepONet()
    print(model)
    u = torch.randn(20, 100)
    y = torch.randn(20, 1)
    outputs = model(u, y)
    print(outputs)
