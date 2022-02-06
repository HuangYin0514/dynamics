# % number of realizations to generate
import numpy as np

from gen_data.burger.GRF import GRF

N = 10

# % parameters for the Gaussian random field
gamma = 4
tau = 5
sigma = 25 ** 2

# % viscosity
visc = 0.01

# % grid size
s = 4096
steps = 100
nn = 101

input = np.zeros((N, nn))
if steps == 1 :
    output = np.zeros((N, s))
else:
    output = np.zeros((N, steps, nn))

tspan = np.linspace(0,1,steps+1)
x = np.linspace(0,1, s+1)
X = np.linspace(0,1, nn)




u0 = GRF(s/2, 0, gamma, tau, sigma, "periodic")