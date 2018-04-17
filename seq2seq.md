# Seq2seq模型学习
https://tensorflow.google.cn/tutorials/seq2seq#inference_how_to_generate_translations

本次学习基本以阅读此篇tutorials为主，实现古龙风格的闲聊机器人，作为代码实现练习。

古龙风格闲聊机器人代码地址：https://github.com/gxdalu-yaya/gulong-chat


![encoder-decoder](images/encdec.jpg)

如上图所示，首先将要翻译的句子进行编码，得到一个中间语义向量，然后用一个decoder，得到翻译结果。


## seq2seq模型train&&inference

![Neural machine translation](images/seq2seq.jpg)

和其他模型不一样，seq2seq模型在训练的时候，如上图所示，有target input words作为输入，这个target input words要用两次，一次是要做embedding后作为decoder的输入，一次是在计算损失函数的时候，作为真实标签，不过严格意义上这两个不一样，作为输入时，在前面加了个< s >作为开始符，在作为真实标签时，在后面加了一个< /s >作为结束符。

但是在预测的时候，是没有target input words的，因此需要首先给一个< s >作为输入，剩下的就用模型预测出来的作为输入。









