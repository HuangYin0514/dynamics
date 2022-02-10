from functools import partial

import jax.numpy as np
import scipy.io
import scipy.io
from jax import random, vmap, jit
from torch.utils import data

if __name__ == '__main__':
    rng_key=random.PRNGKey(1234)
    key, subkey = random.split(rng_key)
    print(key)
    print(subkey)