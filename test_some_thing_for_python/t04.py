from math import sin
import numpy as np
import torch
from scipy.integrate import odeint

g = 9.8

def pendulum_equations(w, t, l):
    th, v = w
    dth = v
    dv = - g/l * sin(th)
    return dth, dv

if __name__ == "__main__":
    print( torch.acos(torch.zeros(1)).item() * 2)