#### 交叉熵损失函数和MSE损失函数区别以及优缺点

理想情况下，我们希望神经网络模型可以从一些错误中快速学习，并改正错误。

MSE损失函数的缺点是，相对于交叉熵损失函数而言，学习的比较慢。

##### MSE损失函数学习慢的原因
而学习慢的原因在于，学习的过程中，要不断的更新w和b，而w和b的更新程度，取决于w和b对于损失函数的偏导数(partial derivatives),即$\partial C/\partial w$和$\partial C/\partial b$ 。所以学习的慢，在一定程度上，就是说偏导数比较小。

MSE的损失函数为:
$$C=\frac{(y-a)^2}{2}$$

其中，a即为神经网络的输出, $a=\sigma(z)$， 而$z=wx+b$

通过求导，得到(假设输入为1，x=1,输出为0，y=0)：
$$\frac{\partial C}{\partial w} = (a-y)\sigma\prime(z) = a\sigma\prime(z)$$
$$\frac{\partial C}{\partial b} = (a-y)\sigma\prime(z) = a\sigma\prime(z)$$

由于偏导数中存在sigmoid函数的导数，而sigmoid的导数在两边时，会趋近于0，所以$\partial C/\partial w$和$\partial C/\partial b$ 也会比较小，导致学习的速度很慢。







