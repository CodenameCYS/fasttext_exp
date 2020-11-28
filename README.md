# fasttext_exp

这一项目用于对fasttext模型做一些基本的测试实验

<!-- TOC -->

- [fasttext_exp](#fasttext_exp)
    - [1. 数据文件](#1-数据文件)
    - [2. 数据处理脚本](#2-数据处理脚本)
    - [3. fastText实验](#3-fasttext实验)
    - [4. tensorflow实验](#4-tensorflow实验)
    - [5. pytorch实验](#5-pytorch实验)
    - [6. 不同cross entropy定义下的模型收敛性实验](#6-不同cross-entropy定义下的模型收敛性实验)

<!-- /TOC -->

## 1. 数据文件

这里，我们采用imdb电影评论打分数据作为我们的训练以及测试语料。

我们将已下载好的imdb数据包放于[data目录](https://github.com/CodenameCYS/fasttext_exp/tree/main/data)下，使用时请自行解压。

## 2. 数据处理脚本

给出数据处理脚本如下：
- [data_processor.py](https://github.com/CodenameCYS/fasttext_exp/blob/main/data_processor.py)

该文件生成fasttext与tensorflow的训练数据，分别存储与[fasttext](https://github.com/CodenameCYS/fasttext_exp/tree/main/data/fasttext)与[tensorflow](https://github.com/CodenameCYS/fasttext_exp/tree/main/data/tensorflow)目录下。

## 3. fastText实验

使用fasttext库进行的fasttext分类模型训练的代码为：
- [fastText_exp.py](https://github.com/CodenameCYS/fasttext_exp/blob/main/fastText_exp.py)

实验结果保存于[fasttext_exp.log](https://github.com/CodenameCYS/fasttext_exp/blob/main/log/fasttext_exp.log)文件下。

## 4. tensorflow实验

使用tensorflow自行写作的fasttext分类模型训练的代码为：
- [tensorflow_exp.py](https://github.com/CodenameCYS/fasttext_exp/blob/main/tensorflow_exp.py)

实验结果保存于[tensorflow_exp.log](https://github.com/CodenameCYS/fasttext_exp/blob/main/log/tensorflow_exp.log)文件下。

## 5. pytorch实验

使用pytorch自行写作的fasttext分类模型训练的代码为：
- [pytorch_exp.py](https://github.com/CodenameCYS/fasttext_exp/blob/main/pytorch_exp.py)

实验结果保存于[pytorch_exp.log](https://github.com/CodenameCYS/fasttext_exp/blob/main/log/pytorch_exp.log)文件下。

## 6. 不同cross entropy定义下的模型收敛性实验

这里，我们考虑两种“cross entropy”定义下模型的收敛性实验。

其中，两种定义分别如下：

1. 真实的cross entropy

    $$L = -\sum_{i}p(x_i) \cdot log(q(x_i))$$

2. 虚假的cross entropy

    $$L = -\sum_{i}(p(x_i) \cdot log(q(x_i)) + (1-p(x_i)) \cdot log(1- q(x_i)))$$

实验代码如下：
1. [pytorch_exp_v2_1.py](https://github.com/CodenameCYS/fasttext_exp/blob/main/pytorch_exp_v2_1.py)
2. [pytorch_exp_v2_2.py](https://github.com/CodenameCYS/fasttext_exp/blob/main/pytorch_exp_v2_2.py)

实验结果显示在notebook文件[模型效果测试.ipynb](https://github.com/CodenameCYS/fasttext_exp/blob/main/%E6%A8%A1%E5%9E%8B%E6%95%88%E6%9E%9C%E6%B5%8B%E8%AF%95.ipynb)当中。

更具体的cross entropy分析详见我的相关博客：[NLP笔记：浅谈交叉熵（cross entropy）](https://blog.csdn.net/codename_cys/article/details/110288295)。