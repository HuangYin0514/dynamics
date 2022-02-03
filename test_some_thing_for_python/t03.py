import torch
import torch.nn as nn

if __name__ == "__main__":
    in_dim = 20

    mu = torch.arange(in_dim).float() / in_dim * 5.0
    mu = nn.Parameter(mu, requires_grad=False)
    print()
