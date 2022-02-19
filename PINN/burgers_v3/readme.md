# 版本特性

- pinn burgers v3

修改

# 项目贡献

1. 项目主要利用PINNs，对Burgers方程进行求解。

Burgers方程：

![](md_file/bugers_equation.png)

2. 重构项目代码。主要优化项目结构，但是部分代码还存在耦合情况，如predict.py与plot_result.py
3. 数据集。数据集主要采用 [pinns](https://github.com/maziarraissi/PINNs.git) 中的数据集
4. 精度

|       模式       | 精度       |
|:--------------:|----------|
| 增加ic和bc采样点数目（n_ic->100,n_bc->100） | 2.59e-03   | 

结果可以参考文件[deeponets.ipynb](result/deeponets.ipynb)

5. 总结
    1. 增加ic和bc采样点数目，精度下降