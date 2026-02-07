# 掩码生成函数
import torch

def create_padding_mask(src, tgt, pad_idx=0):
    # 生成编码器和解码器中的多头注意力层用到的mask
    # 编码器中的mask是为了屏蔽padding
    # 这个padding会出现在src_seq和tgt_seq中较短的一个中，用来统一两个序列的长度
    # pad_idx代表：在序列中代表padding位置的是0

    # src和tgt的形状
    # src: (batch_size, src_seq_len)
    # tgt: (batch_size, tgt_seq_len)

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    # (src != pad_idx)和(tgt != pad_idx)都是由0和1组成的张量
    # 两个unsqueeze是扩充mask的维度，以匹配后续操作
    # .unsqueeze(1).unsqueeze(2): 添加第1和第2维度，扩充后形状为(batch_size, 1, 1, src_seq_len)
    # .unsqueeze(1).unsqueeze(3)：添加第1/3维度，扩充后形状为(batch_size, 1, tgt_seq_len, 1)

    tgt_len = tgt.size(1)
    look_ahead_mask = torch.ones(tgt_len, tgt_len).tril().bool().unsqueeze(0).unsqueeze(0)
    # 创建解码器中多头注意力层的上三角掩码

    tgt_mask = tgt_mask & look_ahead_mask.to(tgt.device)
    # 解码器中的完整掩码由因果掩码和padding掩码拼接组成
    # 在使用多gqu时，可能会出现两个张量存储在不同设备上的情况，所以需要to(tgt.device)使两个张量在同一个设备上以实现拼接操作

    return src_mask, tgt_mask
