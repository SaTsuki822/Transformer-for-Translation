# 前馈神经网络模块
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model：模型的隐藏维度（通常为512、768、1024等）
        # d_ff：前馈网络的中间维度（通常为d_model的4倍，如2048）
        # dropout：防止过拟合的丢弃率，默认为0.1
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        # 第一个全连接层，将维度从d_model扩展到d_ff

        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(d_ff, d_model)
        # 第二个线性层，将维度映射回去

        self.activation = nn.ReLU()

    def forward(self, x):
        # 输入形状：(batch_size, seq_len, d_model)

        x = self.linear1(x)
        # 扩展维度

        x = self.activation(x)
        # 应用激活函数

        x = self.dropout(x)

        x = self.linear2(x)
        # 恢复维度

        return x

