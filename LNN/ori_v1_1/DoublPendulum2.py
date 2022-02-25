from functools import partial  # reduces arguments to function by making some subset implicit
import jax

import jax
import jax.numpy as jnp
import numpy as np # get rid of this eventually

from physics import analytical_fn
from data import get_trajectory_analytic

# vfnc = jax.jit(jax.vmap(analytical_fn))
vget = partial(jax.jit, backend='cpu')(jax.vmap(partial(get_trajectory_analytic, mxsteps=100), (0, None), 0))
