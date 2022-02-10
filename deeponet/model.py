import itertools
from functools import partial

import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.experimental import optimizers
from jax.flatten_util import ravel_pytree
from jax.nn import relu


# Define MLP
def MLP(layers, activation=relu):
    ''' Vanilla MLP'''

    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply


# Define modified MLP
def modified_MLP(layers, activation=relu):
    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b

    def init(rng_key):
        U1, b1 = xavier_init(random.PRNGKey(12345), layers[0], layers[1])
        U2, b2 = xavier_init(random.PRNGKey(54321), layers[0], layers[1])

        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init(k1, d_in, d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return (params, U1, b1, U2, b2)

    def apply(params, inputs):
        params, U1, b1, U2, b2 = params
        U = activation(np.dot(inputs, U1) + b1)
        V = activation(np.dot(inputs, U2) + b2)
        for W, b in params[:-1]:
            outputs = activation(np.dot(inputs, W) + b)
            inputs = np.multiply(outputs, U) + np.multiply(1 - outputs, V)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply


# Define Physics-informed DeepONet model
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = modified_MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = modified_MLP(trunk_layers, activation=np.tanh)

        # Initialize
        branch_params = self.branch_init(rng_key=random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key=random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
                                                                                                       decay_steps=2000,
                                                                                                       decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # Define DeepONet architecture
    def operator_net(self, params, u, t, x):
        branch_params, trunk_params = params
        y = np.stack([t, x])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return outputs

    # Define ds/dx
    def s_x_net(self, params, u, t, x):
        s_x = grad(self.operator_net, argnums=3)(params, u, t, x)
        return s_x

    # Define PDE residual
    def residual_net(self, params, u, t, x):
        s = self.operator_net(params, u, t, x)
        s_t = grad(self.operator_net, argnums=2)(params, u, t, x)
        s_x = grad(self.operator_net, argnums=3)(params, u, t, x)
        s_xx = grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, t, x)

        res = s_t + s * s_x - 0.01 * s_xx
        return res

    # Define initial loss
    def loss_ics(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])

        # Compute loss
        loss = np.mean((outputs.flatten() - s_pred) ** 2)
        return loss

    # Define boundary loss
    def loss_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s_bc1_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])
        s_bc2_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:, 2], y[:, 3])

        s_x_bc1_pred = vmap(self.s_x_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])
        s_x_bc2_pred = vmap(self.s_x_net, (None, 0, 0, 0))(params, u, y[:, 2], y[:, 3])

        # Compute loss
        loss_s_bc = np.mean((s_bc1_pred - s_bc2_pred) ** 2)
        loss_s_x_bc = np.mean((s_x_bc1_pred - s_x_bc2_pred) ** 2)

        return loss_s_bc + loss_s_x_bc

    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])

        # Compute loss
        loss = np.mean((outputs.flatten() - pred) ** 2)
        return loss

        # Define total loss

    def loss(self, params, ics_batch, bcs_batch, res_batch):
        loss_ics = self.loss_ics(params, ics_batch)
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = 20 * loss_ics + loss_bcs + loss_res
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ics_batch, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, ics_batch, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, ics_dataset, bcs_dataset, res_dataset, nIter=10000):
        ics_data = iter(ics_dataset)
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        # pbar = trange(nIter)
        # Main training loop
        for it in range(nIter):
            # Fetch data
            ics_batch = next(ics_data)
            bcs_batch = next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, ics_batch, bcs_batch, res_batch)

            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, ics_batch, bcs_batch, res_batch)
                loss_ics_value = self.loss_ics(params, ics_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                # pbar.set_postfix({'Loss': loss_value,
                #                   'loss_ics': loss_ics_value,
                #                   'loss_bcs': loss_bcs_value,
                #                   'loss_physics': loss_res_value})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, U_star, Y_star[:, 0], Y_star[:, 1])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, Y_star[:, 0], Y_star[:, 1])
        return r_pred


if __name__ == '__main__':
    # Initialize model
    m = 101  # number of sensors for input samples
    branch_layers = [m, 100, 100, 100, 100, 100, 100, 100]
    trunk_layers = [2, 100, 100, 100, 100, 100, 100, 100]
    model = PI_DeepONet(branch_layers, trunk_layers)
