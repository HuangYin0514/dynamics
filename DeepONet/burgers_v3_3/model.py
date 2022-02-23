import torch
import torch.nn as nn

from model_layers.branch_net import BranchNet
from model_layers.trunk_net import trunk_net


class DeepONet(nn.Module):
    """
        Deep operator network.
        Input: ([batch size, branch_dim], [batch size, trunk_dim])
        Output: [batch size, 1]
    """

    def __init__(self):
        super().__init__()
        self.branch_net = BranchNet()
        self.trunk_net = trunk_net()
        self.net_bias = nn.Parameter(torch.zeros([1]))

        # self.branch_net.apply(weights_init_xavier_normal)
        # self.trunk_net.apply(weights_init_xavier_normal)

    def forward(self, u, y):
        B = self.branch_net(u)
        T = self.trunk_net(y)
        outputs = torch.sum(B * T, dim=1) + self.net_bias
        return outputs[:, None]


if __name__ == '__main__':
    model = DeepONet()
    print(model)
    u = torch.randn(20, 101)
    y = torch.randn(20, 2)
    outputs = model(u, y)
    print(outputs)
