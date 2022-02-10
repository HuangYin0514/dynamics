import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from torch.utils.data import Dataset


if __name__ == '__main__':
    data = scipy.io.loadmat("data/Burger.mat")
    output = data["output"]

    for i in range(10):
        plt.imshow(output[i])
        plt.show()
    print()
