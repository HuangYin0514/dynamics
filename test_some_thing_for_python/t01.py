import numpy as np


if __name__ == "__main__":
    x = np.linspace(0, 1, 256).flatten()[:, None]

    res = 0.04 * x * (1 - x)

    print(res)
    print()