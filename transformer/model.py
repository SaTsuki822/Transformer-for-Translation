# transformer模型定义
from torch import nn
import math
# 载入python库

from transformer.PositionalEncoding import PositionalEncoding
from transformer.Encoder import Encoder
from transformer.Decoder import Decoder
# 载入定义好的类

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 n_heads,
                 d_ff,
                 num_layers,
                 dropout=0.1):
        # src_vocab_size (int): 源语言词汇表大小, 如英语词汇表有30000个token
        # tgt_vocab_size (int): 目标语言词汇表大小
        # d_model: 模型嵌入维度
        # n_heads: 注意力头数
        # d_ff 前馈神经网络中间层维度
        # num_layers 编码器/解码器层数

        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # 编码器/解码器的token嵌入层

        self.positional_encoding = PositionalEncoding(d_model)
        # 位置嵌入层

        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(d_model, n_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, num_layers, dropout)
        # 定义编码器和解码器

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        # 输出线性层定义
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src和tgt的形状：
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)

        src = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)
        # token嵌入部分
        # 缩放因子：math.sqrt(d_model)
        # encoder_embedding.embedding_dim其实就是d_model
        # 作用：保持向量方差在合理范围内，嵌入初始化通常方差较小，需要放大以匹配位置编码的尺度

        src = self.dropout(src)
        tgt = self.dropout(tgt)
        # dropout操作，防止过拟合

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # 添加位置编码

        enc_output = self.encoder(src, src_mask)
        # 计算encoder的输出

        dec_output = self.decoder(tgt, enc_output, tgt_mask)
        # 用tgt和enc_output计算decoder的输出

        output = self.fc_out(dec_output)
        # 最后的线性层，用作输出投影

        return output