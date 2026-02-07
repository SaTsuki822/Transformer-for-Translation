# 解码器
from torch import nn
# 导入python库

from transformer.MHA import MultiHeadAttention
from transformer.FFN import FeedForward
# 导入之前定义好的类

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 第一个子层

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # 第一组dropout和norm

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 第二个子层，用来建立与Encoder的联系

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # 第二组dropout和norm

        self.ffn = FeedForward(d_model, d_ff, dropout)
        # 第三个子层，前馈神经网络层

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        # 第三组dropout和norm
    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        # decoder的第一个子层为带mask的多头自注意力机制

        # 输入形状：
        # tgt: (batch_size, tgt_seq_len, d_model)
        # memory: (batch_size, src_seq_len, d_model)
        # tgt_mask: (batch_size, 1, 1, tgt_seq_len)
        # src_mask: (batch_size, 1, 1, src_seq_len)
        x = tgt
        output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        x = self.norm1(x + self.dropout1(output))
        # 这里的x + self.dropout1(output)就代表残差链接

        output = self.cross_attn(x, src, src, src_mask)
        x = self.norm2(x + self.dropout2(output))

        output = self.ffn(x)
        x = self.norm3(x + self.dropout3(output))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        # 和Encoder里的一样，用列表推导式添加DecoderLayer

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # 这里传入的memory代表的是编码器输出
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        return x