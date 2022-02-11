import numpy as np
from sklearn import gaussian_process as gp
import matplotlib.pyplot as plt
from scipy import interpolate

if __name__ == '__main__':
    x = np.linspace(0, 10, num=11, endpoint=True)
    y = np.cos(-x ** 2 / 9.0)
    f = interpolate.interp1d(x, y)
    f2 = interpolate.interp1d(x, y, kind='cubic')

    plt.figure()
    plt.plot(f(x),'--')
    plt.plot(4.1,f2(4.1),'o')
    plt.plot(y, 'x')
    plt.show()

    print(x)
