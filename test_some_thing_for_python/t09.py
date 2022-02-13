

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import jax.numpy as np
import jax.random as random


if __name__ == '__main__':

    key = random.PRNGKey(2021)
    res = random.choice(key, 10, (20,), replace=False)


