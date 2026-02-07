# 多头注意力
import torch
from torch import nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # d_model: 模型嵌入维度
        # n_heads: 注意力头数
        # dropout: 随机丢弃率
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # d_k: 每个注意力头的维度

        assert{
            self.d_k * n_heads == d_model
        }, f"d_model {d_model} not divisible by n_heads {n_heads}"
        # 通过断言机制确保d_model能被n_heads整除，否则无法均匀分配

        self.W_q = nn.Linear(d_model, d_model, bias = False)
        # 计算q查询的参数矩阵
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        # 计算k键
        self.W_v = nn.Linear(d_model, d_model, bias = False)
        # 计算v值
        # W_q/W_k/W_v这三个都是可更新的参数矩阵

        self.W_o = nn.Linear(d_model, d_model)
        # 模块输出位置的线性变换，将多头注意力结果合并回原始维度

        self.dropout = nn.Dropout(dropout)
        # 定义dropout层

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 计算QK相似性得分(那张插图应该都看过)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            # 应用掩码，这个在解码器中会用到，编码器中用不到

        attn_weights = F.softmax(scores, dim=-1)
        # 将QK的运算结果在最后一个维度进行softmax
        # 将每个查询与所有键的相似度转换为概率分布，确保每个查询的注意力权重和为1，形成有效的加权平均
        # 另外，softmax的指数特性会放大高相似度，抑制低相似度，突出重要关系

        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output

    def forward(self, Q, K, V, mask=None):
        # Q/K/V的输入形状均为(batch_size, seq_len, d_model)

        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 对Q/K/V矩阵进行线性变换+维度重塑
        # 线性变换：(batch, seq_len, d_model) → (batch, seq_len, d_model)
        # view重塑：(batch, seq_len, d_model) → (batch, seq_len, n_heads, d_k)
        # transpose转置：(batch, seq_len, n_heads, d_k) → (batch, n_heads, seq_len, d_k)
        # 计算QKV矩阵后进行维度变换以适配后续的scaled_dot_product_attention操作

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # 计算注意力输出

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # 将多头注意力输出进行合并

        output = self.W_o(attn_output)
        # 进行最终的线性变换

        return output
