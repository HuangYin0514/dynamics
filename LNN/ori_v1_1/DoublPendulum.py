from functools import partial  # reduces arguments to function by making some subset implicit

import jax
import jax.numpy as jnp
import numpy as np  # get rid of this eventually

from data import get_trajectory_analytic
from physics import analytical_fn

vfnc = jax.jit(jax.vmap(analytical_fn))
vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic), (0, None), 0))


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


args = ObjectView({'dataset_size': 200,
                   'fps': 10,
                   'samples': 100,
                   'num_epochs': 80000,
                   'seed': 0,
                   'loss': 'l1',
                   'act': 'softplus',
                   'hidden_dim': 600,
                   'output_dim': 1,
                   'layers': 3,
                   'n_updates': 1,
                   'lr': 0.001,
                   'lr2': 2e-05,
                   'dt': 0.1,
                   'model': 'gln',
                   'batch_size': 512,
                   'l2reg': 5.7e-07,
                   })
# #
rng = jax.random.PRNGKey(args.seed)

vfnc = jax.jit(jax.vmap(analytical_fn, 0, 0))
vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic, mxsteps=100), (0, None), 0))
minibatch_per = 2
batch = 512


@jax.jit
def get_derivative_dataset(rng):
    # randomly sample inputs

    y0 = jnp.concatenate([
        jax.random.uniform(rng, (batch * minibatch_per, 2)) * 2.0 * np.pi,
        (jax.random.uniform(rng + 1, (batch * minibatch_per, 2)) - 0.5) * 10 * 2
    ], axis=1)

    return y0, vfnc(y0)


best_params = None
best_loss = np.inf

from HyperparameterSearch import extended_mlp

init_random_params, nn_forward_fn = extended_mlp(args)
import HyperparameterSearch

HyperparameterSearch.nn_forward_fn = nn_forward_fn
_, init_params = init_random_params(rng + 1, (-1, 4))
rng += 1
model = (nn_forward_fn, init_params)
from jax.experimental import optimizers

opt_init, opt_update, get_params = optimizers.adam(args.lr)
opt_state = opt_init([[l2 / 200.0 for l2 in l1] for l1 in init_params])
from jax.tree_util import tree_flatten
from lnn import raw_lagrangian_eom
from HyperparameterSearch import learned_dynamics


@jax.jit
def loss(params, batch, l2reg):
    state, targets = batch  # _rk4
    leaves, _ = tree_flatten(params)
    l2_norm = sum(jnp.vdot(param, param) for param in leaves)
    preds = jax.vmap(
        partial(
            raw_lagrangian_eom,
            learned_dynamics(params)))(state)
    return jnp.sum(jnp.abs(preds - targets)) + l2reg * l2_norm / args.batch_size


@jax.jit
def update_derivative(i, opt_state, batch, l2reg):
    params = get_params(opt_state)
    param_update = jax.grad(
        lambda *args: loss(*args) / len(batch),
        0
    )(params, batch, l2reg)
    #     param_update = normalize_param_update(param_update)
    params = get_params(opt_state)
    return opt_update(i, param_update, opt_state), params


best_small_loss = np.inf
(nn_forward_fn, init_params) = model
iteration = 0
total_epochs = 1
minibatch_per = 2
train_losses, test_losses = [], []

lr = 1e-5  # 1e-3

final_div_factor = 1e4


# OneCycleLR:
@jax.jit
def OneCycleLR(pct):
    # Rush it:
    start = 0.2  # 0.2
    pct = pct * (1 - start) + start
    high, low = lr, lr / final_div_factor

    scale = 1.0 - (jnp.cos(2 * jnp.pi * pct) + 1) / 2

    return low + (high - low) * scale


from lnn import custom_init

opt_init, opt_update, get_params = optimizers.adam(
    OneCycleLR
)

init_params = custom_init(init_params, seed=0)

opt_state = opt_init(init_params)
# opt_state = opt_init(best_params)
bad_iterations = 0
print(lr)

rng = jax.random.PRNGKey(0)
epoch = 0
batch_data = get_derivative_dataset(rng)[0][:1000], get_derivative_dataset(rng)[1][:1000]
print(batch_data[0].shape)
loss(get_params(opt_state), batch_data, 0.0) / len(batch_data[0])
opt_state, params = update_derivative(0.0, opt_state, batch_data, 0.0)

from copy import deepcopy as copy

for epoch in range(epoch, total_epochs):
    epoch_loss = 0.0
    num_samples = 0
    all_batch_data = get_derivative_dataset(rng)
    for minibatch in range(minibatch_per):
        fraction = (epoch + minibatch/minibatch_per)/total_epochs
        batch_data = (all_batch_data[0][minibatch*batch:(minibatch+1)*batch], all_batch_data[1][minibatch*batch:(minibatch+1)*batch])
        rng += 10
        opt_state, params = update_derivative(fraction, opt_state, batch_data, 1e-6)
        cur_loss = loss(params, batch_data, 0.0)
        epoch_loss += cur_loss
        num_samples += batch
    closs = epoch_loss/num_samples
    print('epoch={} lr={} loss={}'.format(
        epoch, OneCycleLR(fraction), closs)
         )
    if closs < best_loss:
        best_loss = closs
        best_params = [[copy(jax.device_get(l2)) for l2 in l1] if len(l1) > 0 else () for l1 in params]
