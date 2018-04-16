## BasicLSTMCell源码学习
源码地址为：
https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn_cell_impl.py
![LSTM](https://raw.githubusercontent.com/gxdalu-yaya/tensorflow-learning/master/images/LSTM3-focus-f.png)
从第一步开始，当前输入$x_{t}$和上一个cell的隐层$h_{t-1}$，首先需要拼接到一起，对应的源码为：

```
gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
```
其中，self._kernal和self._bias为：
```
self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units])
self._bias = self.add_variable(
    _BIAS_VARIABLE_NAME,
    shape=[4 * self._num_units],
    initializer=init_ops.zeros_initializer(dtype=self.dtype))
```
也就是说，$W_{f}$的shape为[input_depth+hidden_size, 4*hidden_size]，不太明白，为啥bias的长度为4×self._num_units,暂时理解为C的shape为4×hiddend_size

```
# i = input_gate, j = new_input, f = forget_gate, o = output_gate
i, j, f, o = array_ops.split(
    value=gate_inputs, num_or_size_splits=4, axis=one)
```
接下来，把计算得到的结果分成4份，分别对应图中的$f_{t}$,  $i_{t}$, $o_{t}$, $j_{t}$，$j_{t}$为tanh激活函数的输入

输出的new_c和new_h为：
```
new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                multiply(sigmoid(i), self._activation(j)))
new_h = multiply(self._activation(new_c), sigmoid(o))
```
forget_bias_tensor即为BasicLSTMCell的第二个参数，We add forget_bias (default: 1) to the biases of the forget gate in order to reduce the scale of forgetting in the beginning of the training.大概意思就是减小刚开始遗忘的程度。
```
new_state = LSTMStateTuple(new_c, new_h)
```
最终函数返回new_h, new_state





