# 编码器
import torch
from torch import nn
import torch.nn.functional as F
import math
# 载入python库

from transformer.MHA import MultiHeadAttention
from transformer.FFN import FeedForward
# 载入其他python文件定义好的类

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        # d_model: 模型维度（特征维度），如512
        # n_heads: 多头注意力机制的头数，如8
        # d_ff: 前馈网络隐藏层维度，通常比d_model大，如2048
        # dropout: 丢弃率，防止过拟合
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 定义多头自注意力层，作为第一个子层

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # 第一组dropout和layernorm，应用在第一个子层后

        self.ffn = FeedForward(d_model, d_ff, dropout)
        # 定义第二个子层：前馈神经网络

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # 第二组dropout和layernorm

    def forward(self, x, mask=None):
        # 输入形状：(batch_size, seq_len, d_model)

        attn_output = self.self_attn(x, x, x, mask)
        # Q/K/V参数都传入x 因为会有线性层进行处理，生成每个头对应的QKV,这里就不需要再加了
        # 这里在encoder层中的多头注意力中也会添加mask
        # src序列中可能会通过padding来统一系列长度，但padding部分是不能参与注意力计算的
        # 因此在这里需要用mask来标记真正的token位置，只处理真正的token

        x = self.norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(x)
        # 第二个子层，进行前馈神经网络计算

        x = self.norm2(x + self.dropout2(ffn_output))

        return x

# 用上面定义好的EncoderLayer定义Encoder类
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        # 上面这行是python的列表推导式语法
        # 等效写法：
        # layers_list = []
        # for i in range(num_layers):
        #     layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        #     layers_list.append(layer)
        # self.layers = nn.ModuleList(layers_list)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 输入x的形状：(batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x