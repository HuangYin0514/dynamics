# 版本特性

- DeepONets burgers_v2_1

使用DeepOnets数据集


# 项目贡献

1. 项目主要利用DeepONets，对Burgers方程进行求解。

Burgers方程：

![](md_file/bugers_equation.png)

2. 重构项目代码。主要优化项目结构，但是部分代码还存在耦合情况，如predict.py与plot_result.py
3. 数据集。数据集主要采用 [DeepONets](https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets.git) 中的数据集
4. 精度

|           模式            | 精度      |
|:-----------------------:|---------|
| deeponets（burgers_v2_1） | 2.49e-03 |

5. 总结

    1. 提高边缘ic的权重，对结果有促进作用

