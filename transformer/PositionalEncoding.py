# 位置编码
import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    # 继承torch.nn，此类定义为一个神经网络模块
    def __init__(self, d_model = 512, max_len=500):
        # d_model： 模型的嵌入维度
        # max_len: 模型支持的最大序列长度
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 创建从0到maxlen-1的位置索引
        # torch.arange() 是 PyTorch 中用于创建等差序列张量的函数
        # 示例：查看position张量的内容
        # print("------position: ", position)
        # 当max_len = 5时，输出为：------position:  tensor([[0.], [1.], [2.], [3.], [4.]])

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # 频率项计算， 这个变量和tranformer本身无关，是位置编码计算过程中用到的一个变量
        # 第一部分：torch.arange(0, d_model, 2)：
        # 生成[0, 2, 4, ..., d_model-2]， 为什么是2？因为偶数索引用sin，奇数索引用cos
        # 这就是位置编码公式中的频率项：1/10000^(2i/d_model)

        pe = torch.zeros(1, max_len, d_model)
        # 初始化位置编码矩阵
        # 形状为(1, max_len, d_model)，第一个维度是1便于广播

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # 给上面定义好的位置编码矩阵pe赋值
        # 第二个索引":"代表选中这个维度上的所有索引
        # 0::2代表从0开始进行步长为2的索引，选中0，2，4，6...
        # 1:2代表选中1，3，5，7...
        # 后面的是矩阵中填入的位置编码内容

        self.register_buffer('pe', pe)
        # 将pe矩阵注册为模块的缓冲区，代表不参与梯度更新，随模型保存/加载

    def forward(self, x):
        tmp = self.pe[:, :x.size(1), :]
        x = x + self.pe[:, :x.size(1), :]
        # 输入张量x的形状：(batch_size, seq_len, d_model)
        # :x.size(1) 代表取前x.size(1)个位置，也就是前seq_len个位置
        return x

# pe = PositionalEncoding()
# x = torch.ones(3,3)
# pe.forward(x)