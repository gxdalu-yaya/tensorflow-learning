
# 梯度下降算法概览
介绍一下各种梯度下降算法
## 梯度下降算法
* Batch gradient descent（批量梯度下降）
* Stochastic gradient descent(随机梯度下降)

### Batch gradient desent

批量梯度下降，顾名思义，就是直接用整个训练集计算损失函数的梯度，来更新参数:
$$\theta = \theta - \eta . \nabla_\theta J(\theta)$$

**优点**
* 损失函数为凸函数时，可以保证收敛到全局最优值；如果是非凸函数，则只能收敛到局部最优。

**缺点**
* 但是直接计算整个训练集的梯度的话，一是计算的很慢，二是很 耗费内存。
* 不能够在线更新

### Stochastic gradient descent

stochastic gradient descent测试每次只计算一个训练样本$x^{(i)}$和$y^{(i)}$:
$$\theta = \theta - \eta . \nabla_\theta J(\theta; x^{(i)}; y^{(i)})$$

**优点**
* 和Batch gradient desent相比较，stoschastic gradient descent没有冗余的计算
* 可以在线更新

**缺点**
* SGD一会会造成损失的大幅震荡


## Adam优化器

Adam优化器是一个新手可以比较放心使用的优化器，只需要设置初始学习率，一般就可以得到比较满意的训练结果，初始学习率默认是0.001，另外训练的batch_size，也会在一定程度上影响学习率，一般来说，较大的batch_size, 对应较小的学习率。有时候训练时，出现loss比较震荡的情况时，会建议增大batch_size, 一般就是因为这个原因，其实就是变相的减小了学习率。

Adam名字的来源是***adaptive moment estimation***, 即***动态动量***。 Adam优化器是借鉴了AdaGrad和RMSProp，集合了2个算法的优点，并做了一些自己的改进。

Adam也是一个随机梯度下降的优化方法，其本质是对于一个损失函数，损失函数的变量如果可微的话，那么就可以通过梯度下降的方法来求损失函数的最大值或最小值。