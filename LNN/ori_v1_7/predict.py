import random
from functools import partial

import numpy as np
import scipy.io
import torch
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from physics import kinetic_energy
from physics import potential_energy, analytical_fn
from trainer import Trainer
from utils import get_device, to_numpy, to_tensor

device = get_device()


def init_random_state():
    random_seed = 1308
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.deterministic = True  # speed up computation
    torch.backends.cudnn.benchmark = True


init_random_state()


def get_pred_data():
    y0 = np.concatenate([
        np.random.uniform(size=(1, 2)) * 2.0 * np.pi,
        np.random.uniform(size=(1, 2)) * 0.1
    ], axis=1)
    return y0

def baseline_eom(baseline, state, t=None):
    q, q_t = np.split(state, 2)
    q = q % (2 * np.pi)
    q_tt = baseline(q, q_t)
    return np.concatenate([q_t, q_tt])

def analytical_baseline(q, q_t):
    q_tt = analytical_fn(np.concatenate([q, q_t])[None,:])[:,2:4]
    return np.squeeze(q_tt)

def learned_dynamics(q, q_t):
    q_tt = trainer.predict(torch.unsqueeze(to_tensor(q),0),torch.unsqueeze(to_tensor(q_t),0))
    return to_numpy(torch.squeeze(q_tt))

if __name__ == '__main__':
    # data
    fps = 10
    t_span = [0, 10]
    frames = int(fps * (t_span[1] - t_span[0]))

    y0 = get_pred_data()
    y0 = np.squeeze(y0)

    # model
    model = torch.load('result/model_final.pkl').to(device)

    # trainer
    trainer = Trainer()
    trainer.model = model

    true = odeint(partial(baseline_eom, analytical_baseline), y0, np.linspace(0, t_span[1], num=frames))
    pred = odeint(partial(baseline_eom, learned_dynamics), y0, np.linspace(0, t_span[1], num=frames))

    error = np.sum(np.abs(true-pred))

    plt.figure()
    plt.plot(pred[:30, 0], label="y_pred")
    plt.plot(true[:30, 0], label="y_true")
    plt.ylabel(r'$\theta_1$')
    plt.xlabel('Time')
    plt.legend()
    plt.savefig('result/theta_1.png')
    plt.show()

