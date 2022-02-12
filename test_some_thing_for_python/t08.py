

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import jax.numpy as np
import jax.random
if __name__ == '__main__':
    s_test = jax.random.randint(jax.random.PRNGKey(2022),(10,1),minval=0,maxval=20)
    s_pred = jax.random.randint(jax.random.PRNGKey(2023),(10,1),minval=0,maxval=20)
    res = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
    print(res)