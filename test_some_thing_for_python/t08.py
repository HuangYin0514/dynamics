import jax.random as random

if __name__ == '__main__':
    N_train = 9

    key = random.PRNGKey(0)  # use different key for generating test data
    keys = random.split(key, N_train)
