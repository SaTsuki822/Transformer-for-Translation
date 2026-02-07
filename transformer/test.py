# 示例用法
# 用于排查代码里的错误
import torch
# 导入python库

from transformer.model import Transformer
from transformer.create_mask import create_padding_mask
# 导入定义好的类

if __name__ == '__main__':
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    n_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, num_layers, dropout)

    src = torch.randint(0, src_vocab_size, (32, 10))
    tgt = torch.randint(0, tgt_vocab_size, (32, 12))

    src_mask, tgt_mask = create_padding_mask(src, tgt)

    output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    print(output.shape)