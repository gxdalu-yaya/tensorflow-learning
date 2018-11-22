### 过拟合以及正则化

1.L2正则化
	
   L2正则化就是给损失函数加一个对参数的限制值，也叫正则项。正则化的交叉熵损失函数为：
   $$C=-\frac{1}{n}\sum_{xj}[y_j\ln a_j^L+(1-y_j)\ln (1-a_j^L)]+\frac{\lambda}{2n}\sum_ww^2$$
   
   一个通用的写法是：
   $$C = C_0 + \frac{\lambda}{2n}\sum_ww^2$$
   其中，$C_0$为原始的损失函数项， $\lambda$为正则系数，正则项一般不加偏置，因为当输入变化时，偏置不太会影响最终的变化。
   
   正则化的主要目的是学习较小的权重，同时获得较小的损失，而正则系数的作用，就是在两者之间做一个权衡。
   
   通过一个例子，来证明为什么正则化可以防止过拟合：
   求偏导数得到：
   $$\frac{\partial C}{\partial w} = \frac{\partial C_0}{\partial w}+\frac{\lambda}{n}w$$
   $$\frac{\partial C}{\partial w} = \frac{\partial C_0}{\partial b}$$
   那么参数更新时：
   $$b\rightarrow b-\eta\frac{\partial C_0}{\partial b}$$
   $$w\rightarrow w-\eta\frac{\partial C_0}{\partial w}-\frac{\eta\lambda}{n}w$$
$$=(1-\frac{\eta\lambda}{n})w-\eta\frac{\partial C_0}{\partial w}$$
这样一来，每次权重更新时，w都会先乘一个系数：$(1-\frac{\eta\lambda}{n})$, 这个系数一定是小于1的，那么每次 更新时，w都会以一定的比率缩小，也就是实现了所谓的***权重衰减(weight decay)***，然而一直衰减也不会导致w变成0，因为后面的$\eta\frac{\partial C_0}{\partial w}$有可能让w变大。

#### 正则化可以防止过拟合的原因

1.正则化可以让模型更简单

2.正则化之后，小的输入变化，不会导致太大的输出变化

#### L1正则化容易产生稀疏特征的原因
![L1L2](./images/L1L2.png)
如左图所示，考虑只有两个参数的情况，在（$w_1, w_2$）平面上，可以画出目标函数的等高线，而对参数的约束，见图中的黑线，L1的约束形成了一个“菱形”

