from functools import partial  # reduces arguments to function by making some subset implicit

import jax
import jax.numpy as jnp
import numpy as np # get rid of this eventually

from physics import analytical_fn
from data import get_trajectory_analytic

vfnc = jax.jit(jax.vmap(analytical_fn))
vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic, mxsteps=100), (0, None), 0))

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

import pickle as pkl
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
minibatch_per = 2000
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
