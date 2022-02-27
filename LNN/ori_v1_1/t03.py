import jax
import jax.numpy as jnp
import numpy as np  # get rid of this eventually

from data import get_trajectory_analytic

seed = 0
samples = 1
t_span = [0, 2000]
fps = 1
test_split = 0.5
rng = jax.random.PRNGKey(0)
vget = jax.vmap(get_trajectory_analytic, (0, None), 0)
frames = int(fps * (t_span[1] - t_span[0]))
times = jnp.linspace(t_span[0], t_span[1], frames)
y0 = jnp.concatenate([
    jax.random.uniform(rng, (samples, 2)) * 2.0 * np.pi,
    jax.random.uniform(rng + 1, (samples, 2)) * 0.1
], axis=1)
y = vget(y0, times)
# y= get_trajectory_analytic(y0, times, mxstep=100)
