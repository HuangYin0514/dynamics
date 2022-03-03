# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)

import numpy as np
from scipy.integrate import odeint


def kinetic_energy(state, m1=1, m2=1, l1=1, l2=1, g=9.8):
    q, q_dot = np.split(state, 2, axis=1)

    t1, t2 = q[:, 0:1], q[:, 1:2]
    w1, w2 = q_dot[:, 0:1], q_dot[:, 1:2]

    T1 = 0.5 * m1 * (l1 * w1) ** 2
    T2 = 0.5 * m2 * ((l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2))
    T = T1 + T2
    return T


def potential_energy(state, m1=1, m2=1, l1=1, l2=1, g=9.8):
    q, q_dot = np.split(state, 2, axis=1)

    t1, t2 = q[:, 0:1], q[:, 1:2]
    w1, w2 = q_dot[:, 0:1], q_dot[:, 1:2]

    y1 = -l1 * np.cos(t1)
    y2 = y1 - l2 * np.cos(t2)
    V = m1 * g * y1 + m2 * g * y2
    return V


# def kinetic_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8):
#     (t1, t2), (w1, w2) = q, q_dot
#
#     T1 = 0.5 * m1 * (l1 * w1) ** 2
#     T2 = 0.5 * m2 * ((l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2))
#     T = T1 + T2
#     return T
#
#
# # @jit
# def potential_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8):
#     (t1, t2), (w1, w2) = q, q_dot
#
#     y1 = -l1 * np.cos(t1)
#     y2 = y1 - l2 * np.cos(t2)
#     V = m1 * g * y1 + m2 * g * y2
#     return V


# # Double pendulum lagrangian
def lagrangian_fn(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8):
    (t1, t2), (w1, w2) = q, q_dot

    T = kinetic_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8)
    V = potential_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8)
    return T - V


# Double pendulum dynamics via analytical forces taken from Diego's blog
def analytical_fn(state, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
    t1, t2, w1, w2 = state
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
    a2 = (l1 / l2) * np.cos(t1 - t2)
    f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2 ** 2) * np.sin(t1 - t2) - (g / l1) * np.sin(t1)
    f2 = (l1 / l2) * (w1 ** 2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return np.stack([w1, w2, g1, g2])



