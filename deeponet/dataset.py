from functools import partial

import jax.numpy as np
import scipy.io
import scipy.io
from jax import random, vmap, jit
from torch.utils import data


# Geneate ics training data corresponding to one input sample
def generate_one_ics_training_data(key, u0, m=101, P=101):
    t_0 = np.zeros((P, 1))
    x_0 = np.linspace(0, 1, P)[:, None]

    y = np.hstack([t_0, x_0])
    u = np.tile(u0, (P, 1))
    s = u0

    return u, y, s


# Geneate bcs training data corresponding to one input sample
def generate_one_bcs_training_data(key, u0, m=101, P=100):
    t_bc = random.uniform(key, (P, 1))
    x_bc1 = np.zeros((P, 1))
    x_bc2 = np.ones((P, 1))

    y1 = np.hstack([t_bc, x_bc1])  # shape = (P, 2)
    y2 = np.hstack([t_bc, x_bc2])  # shape = (P, 2)

    u = np.tile(u0, (P, 1))
    y = np.hstack([y1, y2])  # shape = (P, 4)
    s = np.zeros((P, 1))

    return u, y, s


# Geneate res training data corresponding to one input sample
def generate_one_res_training_data(key, u0, m=101, P=1000):
    subkeys = random.split(key, 2)

    t_res = random.uniform(subkeys[0], (P, 1))
    x_res = random.uniform(subkeys[1], (P, 1))

    u = np.tile(u0, (P, 1))
    y = np.hstack([t_res, x_res])
    s = np.zeros((P, 1))

    return u, y, s


# Geneate test data corresponding to one input sample
def generate_one_test_data(idx, usol, m=101, P=101):
    u = usol[idx]
    u0 = u[0, :]

    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, P)
    T, X = np.meshgrid(t, x)

    s = u.T.flatten()
    u = np.tile(u0, (P ** 2, 1))
    y = np.hstack([T.flatten()[:, None], X.flatten()[:, None]])

    return u, y, s


# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s

        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx, :]
        y = self.y[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs


def getBurgersEquationDataSet():
    # Load data
    path = 'data/Burger.mat'  # Please use the matlab script to generate data

    data = scipy.io.loadmat(path)
    usol = np.array(data['output'])

    N = usol.shape[0]  # number of total input samples
    N_train = 10  # number of input samples used for training
    N_test = N - N_train  # number of input samples used for test
    m = 101  # number of sensors for input samples
    P_ics_train = 101  # number of locations for evulating the initial condition
    P_bcs_train = 100  # number of locations for evulating the boundary condition
    P_res_train = 2500  # number of locations for evulating the PDE residual
    P_test = 101  # resolution of uniform grid for the test data

    u0_train = usol[:N_train, 0, :]  # input samples
    # usol_train = usol[:N_train,:,:]

    key = random.PRNGKey(0)  # use different key for generating test data
    keys = random.split(key, N_train)

    # Generate training data for inital condition
    u_ics_train, y_ics_train, s_ics_train = vmap(generate_one_ics_training_data, in_axes=(0, 0, None, None))(keys,
                                                                                                             u0_train,
                                                                                                             m,
                                                                                                             P_ics_train)

    u_ics_train = u_ics_train.reshape(N_train * P_ics_train, -1)
    y_ics_train = y_ics_train.reshape(N_train * P_ics_train, -1)
    s_ics_train = s_ics_train.reshape(N_train * P_ics_train, -1)

    # Generate training data for boundary condition
    u_bcs_train, y_bcs_train, s_bcs_train = vmap(generate_one_bcs_training_data, in_axes=(0, 0, None, None))(keys,
                                                                                                             u0_train,
                                                                                                             m,
                                                                                                             P_bcs_train)

    u_bcs_train = u_bcs_train.reshape(N_train * P_bcs_train, -1)
    y_bcs_train = y_bcs_train.reshape(N_train * P_bcs_train, -1)
    s_bcs_train = s_bcs_train.reshape(N_train * P_bcs_train, -1)

    # Generate training data for PDE residual
    u_res_train, y_res_train, s_res_train = vmap(generate_one_res_training_data, in_axes=(0, 0, None, None))(keys,
                                                                                                             u0_train,
                                                                                                             m,
                                                                                                             P_res_train)

    u_res_train = u_res_train.reshape(N_train * P_res_train, -1)
    y_res_train = y_res_train.reshape(N_train * P_res_train, -1)
    s_res_train = s_res_train.reshape(N_train * P_res_train, -1)

    batch_size = 1010

    ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
    bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
    res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

    return ics_dataset, bcs_dataset, res_dataset


if __name__ == '__main__':
    ics_dataset, bcs_dataset, res_dataset = getBurgersEquationDataSet()