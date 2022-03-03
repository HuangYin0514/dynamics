import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from physics import  potential_energy, analytical_fn
from trainer import Trainer
from physics import kinetic_energy
from utils import get_device, to_numpy, to_tensor

device = get_device()

def get_trajectory_analytic(y0, times, **kwargs):
    return odeint(analytical_fn, y0, t=times, rtol=1e-10, atol=1e-10, **kwargs)

def generate_test_data(sameples=1, fps=10, t_span=[0, 100]):
    y0 = np.concatenate([np.random.uniform(size=(sameples, 2)) * 2.0 * np.pi,
                         (np.random.uniform(size=(sameples, 2)) * 0.1)
                         ], axis=1)

    y0 = np.squeeze(y0, 0)

    frames = int(fps * (t_span[1] - t_span[0]))
    times = np.linspace(t_span[0], t_span[1], frames)
    y_true = get_trajectory_analytic(y0, times)
    k_true = kinetic_energy(y_true)
    v_true = potential_energy(y_true)

    return y0,y_true, k_true, v_true

def pred_fn(y0, t=0,trainer=None):
    y0 = to_tensor(y0)
    q_t, q_tt = trainer.predict(torch.unsqueeze(y0,0))
    y1_pred = torch.cat([q_t, q_tt], 1)
    y1_pred = torch.squeeze(y1_pred, 0)
    y1_pred = to_numpy(y1_pred)
    dydt = y1_pred
    return dydt

def get_trajectory_pred(trainer,y0, times, **kwargs):
    return odeint(pred_fn, y0, t=times,args=(trainer,),rtol=1e-10, atol=1e-10, **kwargs)

def get_pred_data(trainer,y0, fps=10, t_span=[0, 100]):
    frames = int(fps * (t_span[1] - t_span[0]))
    times = np.linspace(t_span[0], t_span[1], frames)

    y_pred = get_trajectory_pred(trainer,y0,times)

    k_pred = kinetic_energy(y_pred)
    v_pred = potential_energy(y_pred)
    return y_pred, k_pred, v_pred

if __name__ == '__main__':
    # data
    fps = 10
    t_span = [0, 10]
    y0,y_true, k_true, v_true = generate_test_data(fps=fps,t_span=t_span)

    # model
    model = torch.load('result/model_final.pkl').to(device)
    # model = torch.load('result/model_final.pkl', map_location=torch.device('cpu')).to(device)

    # trainer
    trainer = Trainer()
    trainer.model = model
    y_pred, k_pred, v_pred = get_pred_data(trainer,y0,fps=fps,t_span=t_span)

    plt.figure()
    plt.plot(y_true[:500, 0])
    plt.plot(y_pred[:500, 0])
    plt.ylabel(r'$\theta_1$')
    plt.xlabel('Time')
    plt.savefig('result/theta_1.png')
    plt.show()

    scale = 29.4
    total_true_energy = (k_true + v_true) / scale
    total_pred_energy = (k_pred + v_pred) / scale
    plt.figure()
    plt.plot(total_true_energy)
    plt.plot(total_pred_energy)
    plt.ylabel(r'$total_true_energy$')
    plt.xlabel('Time')
    plt.savefig('result/total_true_energy.png')
    # plt.ylim(-30, 30)
    plt.show()

