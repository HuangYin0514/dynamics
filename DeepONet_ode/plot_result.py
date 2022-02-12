import numpy as np
import torch
from matplotlib import pyplot as plt

from trainer import Trainer
from utils import get_device, to_numpy, to_tensor

device = get_device()


def plot(trainer):
    x = np.linspace(0, 1, num=100)
    y = np.linspace(0, 1, num=100)

    out = x - np.cos(2 * np.pi * x)
    antide_true = x * x / 2 - np.sin(2 * np.pi * y) / (2 * np.pi)

    u_star = np.tile(out, (len(y), 1))
    y_star = y[:, None]
    u_star, y_star = to_tensor(u_star), to_tensor(y_star)

    antide_pred = trainer.predict(u_star, y_star)
    antide_pred = to_numpy(antide_pred)

    plt.figure(figsize=(20, 8))
    plt.plot(x, out, color='black', label=r'Input: $x-\cos(2\pi x)$', zorder=0)
    plt.plot(y, antide_true, color='b', label=r'Output: $x^2/2 - \sin(2\pi x)/(2\pi)$', zorder=1)
    plt.plot(y, antide_pred, color='r', label=r'Prediction', zorder=2)
    plt.legend()
    plt.savefig('result/deeponet.png')
    # plt.show()


if __name__ == '__main__':
    model = torch.load('result/model_final.pkl').to(device)
    trainer = Trainer()
    trainer.model = model
    plot(trainer)
