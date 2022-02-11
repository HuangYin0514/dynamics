import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos, linspace, pi
from scipy.integrate import odeint, solve_bvp, solve_ivp
import numpy as np



if __name__ == '__main__':
    def u_func (t):

        return  cos(t ** 2)
    def f2(t, y):
        '''
        在scipy1.1.0版本中odeint引进tfirst参数，其值为True时，func的参数顺序为 t,y
        '''
        return u_func(t)


    def solve_first_order_ode():
        '''
        求解一阶ODE
        '''
        t1 = linspace(-10, 10, 1000)
        y0 = [10]  # 为了兼容solve_ivp函数，这里初值要array类型

        y3 = solve_ivp(f2, (-10.0, 10.0), y0, method='LSODA', t_eval=t1)  # 注意参数t_span和t_eval的赋值


        plt.subplot(223)
        plt.plot(y3.t, y3.y[0], 'g', label='solve_ivp')
        plt.legend()
        plt.show()


    solve_first_order_ode()


