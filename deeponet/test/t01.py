
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    test_num = np.random.rand(10,1)
    print(test_num)
    plt.figure()
    plt.plot(test_num)
    plt.show()