import random

import numpy as np



if __name__ == "__main__":
    a = np.array(np.arange(1,10).reshape(3,3))
    b = np.array(np.arange(1,10).reshape(3,3))
    b1 = np.array(np.arange(1, 10).reshape(3, 3))
    c =np.vstack((a, b,b1))
    print(c)
