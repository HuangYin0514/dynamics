import random
from functools import partial

import numpy as np
import scipy.io
import torch
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from physics import analytical_fn
from utils import get_device

device = get_device()


def init_random_state():
    random_seed = 3404
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.deterministic = True  # speed up computation
    torch.backends.cudnn.benchmark = True


init_random_state()

if __name__ == '__main__':

    sameples = 1
    y0 = np.concatenate([np.random.uniform(size=(sameples, 2)) * 2.0 * np.pi,
                         (np.random.uniform(size=(sameples, 2)) * 0.1)
                         ], axis=1)

    y0 = np.squeeze(y0, 0)

    fps = 10
    t_span = [0, 10]
    frames = int(fps * (t_span[1] - t_span[0]))

    q_tt_map = {}

    q_list = []


    def baseline_eom(baseline, state, t=None):
        q, q_t = np.split(state, 2)
        q = q % (2 * np.pi)
        q_tt = baseline(q, q_t)
        return np.concatenate([q_t, q_tt])


    def analytical_baseline(q, q_t):
        q_tt = analytical_fn(np.concatenate([q, q_t]))[2:4]
        q_tt_map[str(q) + str(q_t)] = q_tt
        q_list.append(np.concatenate([q, q_t, q_tt]))
        return q_tt


    def learned_dynamics(q, q_t):
        try:
            q_tt = q_tt_map[str(q) + str(q_t)]
        except:
            q_tt = analytical_fn(np.concatenate([q, q_t]))[2:4]
        return q_tt


    # Double pendulum dynamics via analytical forces taken from Diego's blog
    true = odeint(partial(baseline_eom, analytical_baseline), y0, np.linspace(0, t_span[1], num=frames))
    pred = odeint(partial(baseline_eom, learned_dynamics), y0, np.linspace(0, t_span[1], num=frames))

    plt.figure()
    plt.plot(pred[:500, 0], label="y_pred")
    plt.plot(true[:500, 0], label="y_true")
    plt.ylabel(r'$\theta‘_1$ error')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

    q_list = np.array(q_list)
    scipy.io.savemat('data/double_pendulum_data.mat', {'y0': y0, 'q_list': q_list})
