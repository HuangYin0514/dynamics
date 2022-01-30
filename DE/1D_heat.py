import matplotlib.pyplot as plt
import numpy as np

h = 0.1  # 空间步长
N = 30  # 空间步数
dt = 0.0001  # 时间步长
M = 10000  # 时间的步数
A = dt / (h ** 2)  # lambda*tau/h^2
U = np.zeros([N + 1, M + 1])  # 建立二维空数组
Space = np.arange(0, (N + 1) * h, h)  # 建立空间等差数列，从0到3，公差是h

# 边界条件
for k in np.arange(0, M + 1):
    U[0, k] = 0.0
    U[N, k] = 0.0

# 初始条件
for i in np.arange(0, N):
    U[i, 0] = 14 * i * h * (3 - i * h)

# 递推关系
for k in np.arange(0, M):
    for i in np.arange(1, N):
        U[i, k + 1] = A * U[i + 1, k] + (1 - 2 * A) * U[i, k] + A * U[i - 1, k]

# 不同时刻的温度随空间坐标的变化
plt.figure()
plt.plot(Space, U[:, 0], 'g-', label='t=0', linewidth=1.0)
plt.plot(Space, U[:, 3000], 'b-', label='t=3/10', linewidth=1.0)
plt.plot(Space, U[:, 6000], 'k-', label='t=6/10', linewidth=1.0)
plt.plot(Space, U[:, 9000], 'r-', label='t=9/10', linewidth=1.0)
plt.plot(Space, U[:, 10000], 'y-', label='t=1', linewidth=1.0)
plt.ylabel('u(x,t)', fontsize=20)
plt.xlabel('x', fontsize=20)
plt.xlim(0, 3)
plt.ylim(-2, 10)
plt.legend(loc='upper right')
plt.show()

# plt.figure()
# plt.imshow(U)
# plt.xlim(9000, 9100)
# plt.xlabel("t")
# plt.ylim(0, 30)
# plt.ylabel("x")
# plt.show()

#温度等高线随时空坐标的变化，温度越高，颜色越偏红
extent = [0,1,0,3]#时间和空间的取值范围
levels = np.arange(U.min(),U.max(),0.1)#温度等高线的变化范围0-10，变化间隔为0.1
plt.contourf(U,levels,origin='lower',extent=extent,cmap=plt.cm.jet)
plt.ylabel('x', fontsize=20)
plt.xlabel('t', fontsize=20)
plt.show()