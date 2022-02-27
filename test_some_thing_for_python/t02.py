import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


def diff(y=1, x=2):
    return np.array(x+1)


# 上面定义的函数在odeint里面体现的就是dy/dx = x
x = np.linspace(0, 10, 100)  # 给出x范围
y = odeint(diff, 0, x)  # 设初值为0 此时y为一个数组，元素为不同x对应的y值
# 也可以直接y = odeint(lambda y, x: x, 0, x)
plt.plot(x, y[:, 0])  # y数组（矩阵）的第一列，（因为维度相同，plt.plot(x, y)效果相同）
plt.grid()
plt.show()
