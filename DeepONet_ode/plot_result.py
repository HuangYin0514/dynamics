import numpy as np
import torch
from matplotlib import pyplot as plt

from trainer import Trainer


def plot(trainer):
    x = np.linspace(0, 1, num=100)
    y = np.linspace(0, 1, num=100)

    out = x - np.cos(2 * np.pi * x)
    antide_true = x * x / 2 - np.sin(2 * np.pi * y) / (2 * np.pi)

    u_star = np.tile(out, (len(y), 1))
    y_star = y[:, None]
    u_star = torch.tensor(u_star, dtype=torch.float)
    y_star = torch.tensor(y_star, dtype=torch.float)
    # antide_pred = trainer.predict([np.tile(out, (len(y), 1)), y[:, None]], returnnp=True).squeeze()
    antide_pred = trainer.predict(u_star, y_star).detach().numpy()

    plt.figure(figsize=(20, 8))
    plt.plot(x, out, color='black', label=r'Input: $x-\cos(2\pi x)$', zorder=0)
    plt.plot(y, antide_true, color='b', label=r'Output: $x^2/2 - \sin(2\pi x)/(2\pi)$', zorder=1)
    plt.plot(y, antide_pred, color='r', label=r'Prediction', zorder=2)
    plt.legend()
    plt.savefig('result/deeponet.pdf')
    plt.show()


if __name__ == '__main__':
    model = torch.load('result/model_final.pkl')
    trainer = Trainer()
    trainer.model = model
    plot(trainer)
