# inference.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import yaml
import time
from tqdm import tqdm
import math
from transformer.model import Transformer
from transformer.create_mask import create_padding_mask


# 设置随机种子以保证可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 数据集类（推理用）
class InferenceDataset:
    def __init__(self, data_path, src_vocab, max_len=100):
        """
        读取TSV格式的推理数据集
        格式：英文句子（只有一列）
        """
        try:
            # 尝试读取两列数据
            self.data = pd.read_csv(data_path, sep='\t', header=None)
            # 如果只有一列，则只有源语言
            if len(self.data.columns) == 1:
                self.data.columns = ['src']
                self.has_reference = False
            else:
                self.data.columns = ['src', 'ref']
                self.has_reference = True
        except Exception as e:
            print(f"Error reading file: {e}")
            raise

        self.src_vocab = src_vocab
        self.max_len = max_len

        # 统计信息
        print(f"Loaded {len(self.data)} samples from {data_path}")
        if self.has_reference:
            print("Dataset contains reference translations")
        else:
            print("Dataset contains only source sentences")

    def __len__(self):
        return len(self.data)

    def get_item(self, idx):
        src_sentence = str(self.data.iloc[idx]['src'])

        # 将句子转换为索引序列
        src_indices = self.sentence_to_indices(src_sentence, self.src_vocab)

        # 添加特殊标记：<sos> 和 <eos>
        src_indices = [self.src_vocab['<sos>']] + src_indices + [self.src_vocab['<eos>']]

        # 截断或填充到最大长度
        src_indices = src_indices[:self.max_len]
        src_indices = self.pad_sequence(src_indices, self.max_len, self.src_vocab['<pad>'])

        # 如果有参考翻译，也返回
        if self.has_reference:
            ref_sentence = str(self.data.iloc[idx]['ref'])
            return src_sentence, ref_sentence, torch.tensor(src_indices, dtype=torch.long).unsqueeze(0)
        else:
            return src_sentence, None, torch.tensor(src_indices, dtype=torch.long).unsqueeze(0)

    def sentence_to_indices(self, sentence, vocab):
        """将句子转换为索引序列"""
        # 这里假设句子已经分词，以空格分隔
        tokens = sentence.strip().split()
        indices = []
        for token in tokens:
            if token in vocab:
                indices.append(vocab[token])
            else:
                indices.append(vocab['<unk>'])
        return indices

    def pad_sequence(self, sequence, max_len, pad_idx):
        """填充序列到指定长度"""
        if len(sequence) < max_len:
            sequence = sequence + [pad_idx] * (max_len - len(sequence))
        return sequence[:max_len]


# 贪心解码
def greedy_decode(model, src, src_mask, max_len, start_token, end_token, device):
    """使用贪心算法生成翻译"""
    batch_size = src.size(0)

    # 编码器输出
    memory = model.encoder(
        model.encoder_embedding(src) * math.sqrt(model.encoder_embedding.embedding_dim),
        src_mask
    )

    # 解码器输入初始化（仅包含 <sos> 标记）
    decoder_input = torch.zeros(batch_size, 1, dtype=torch.long).fill_(start_token).to(device)

    # 自回归生成
    for _ in range(max_len - 1):
        # 创建解码器掩码
        _, decoder_mask = create_padding_mask(
            decoder_input,
            decoder_input,
            pad_idx=0
        )

        # 前向传播
        output = model.decoder(
            model.decoder_embedding(decoder_input) * math.sqrt(model.decoder_embedding.embedding_dim),
            memory,
            decoder_mask,
            src_mask
        )

        # 获取下一个词的概率分布
        output = model.fc_out(output)
        next_word = output[:, -1, :].argmax(dim=-1, keepdim=True)

        # 添加到解码器输入
        decoder_input = torch.cat([decoder_input, next_word], dim=1)

        # 如果全部生成了 <eos>，则停止
        if (next_word == end_token).all():
            break

    return decoder_input


# 束搜索解码
def beam_search_decode(model, src, src_mask, max_len, start_token, end_token,
                       beam_width, device, length_penalty=0.6):
    """使用束搜索算法生成翻译"""
    batch_size = src.size(0)

    # 编码器输出
    memory = model.encoder(
        model.encoder_embedding(src) * math.sqrt(model.encoder_embedding.embedding_dim),
        src_mask
    )

    # 初始化束
    beams = [(torch.tensor([[start_token]], device=device), 0.0)]  # (sequence, score)

    for step in range(max_len - 1):
        all_candidates = []

        for seq, score in beams:
            # 如果序列以<eos>结束，则直接加入候选列表
            if seq[0, -1] == end_token:
                all_candidates.append((seq, score))
                continue

            # 创建解码器掩码
            _, decoder_mask = create_padding_mask(seq, seq, pad_idx=0)

            # 前向传播
            output = model.decoder(
                model.decoder_embedding(seq) * math.sqrt(model.decoder_embedding.embedding_dim),
                memory,
                decoder_mask,
                src_mask
            )

            # 获取下一个词的概率分布
            output = model.fc_out(output)
            log_probs = torch.log_softmax(output[:, -1, :], dim=-1)

            # 获取top-k个候选
            topk_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

            for i in range(beam_width):
                candidate_seq = torch.cat([seq, topk_indices[0, i].unsqueeze(0).unsqueeze(0)], dim=1)
                candidate_score = score + topk_probs[0, i].item()

                # 长度惩罚
                length = candidate_seq.size(1)
                candidate_score = candidate_score / (length ** length_penalty)

                all_candidates.append((candidate_seq, candidate_score))

        # 选择top beam_width个候选
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_width]

        # 如果所有束都以<eos>结束，则停止
        if all(beam[0][0, -1] == end_token for beam in beams):
            break

    # 返回最佳序列
    best_seq, best_score = beams[0]
    return best_seq

def load_vocab(filepath):
    """从文件加载词汇表"""
    vocab = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            vocab[word] = int(idx)
    print(f"Loaded vocabulary from {filepath}, size: {len(vocab)}")
    return vocab

# 主推理函数
def main(config_path):
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    set_seed(config['inference']['seed'])

    # 设备设置
    device = torch.device(config['inference']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型检查点
    checkpoint_path = config['inference']['model_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取词汇表和配置
    vocab_dir = config['data'].get('vocab_dir', 'vocab')
    src_vocab_path = os.path.join(vocab_dir, f"{config['data']['src_lang']}_vocab.txt")
    tgt_vocab_path = os.path.join(vocab_dir, f"{config['data']['tgt_lang']}_vocab.txt")
    
    src_vocab = load_vocab(src_vocab_path)
    tgt_vocab = load_vocab(tgt_vocab_path)

    saved_config = checkpoint.get('config', {})

    # 反转目标词汇表用于解码
    idx_to_word_tgt = {idx: word for word, idx in tgt_vocab.items()}

    # 创建模型
    print("Initializing model...")
    model_config = saved_config.get('model', config['model'])

    model = Transformer(
        src_vocab_size=model_config['src_vocab_size'],
        tgt_vocab_size=model_config['tgt_vocab_size'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")

    # 创建数据集
    print("Loading inference data...")
    dataset = InferenceDataset(
        config['inference']['test_path'],
        src_vocab,
        max_len=config['inference']['max_len']
    )

    # 创建输出目录
    output_dir = config['inference']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 获取解码方法
    decode_method = config['inference'].get('decode_method', 'greedy')
    beam_width = config['inference'].get('beam_width', 5)

    print(f"Using {decode_method} decoding method")
    if decode_method == 'beam_search':
        print(f"Beam width: {beam_width}")

    # 推理循环
    print("Starting inference...")
    start_time = time.time()

    results = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Translating"):
            src_sentence, ref_sentence, src_tensor = dataset.get_item(i)

            # 创建源语言掩码
            src = src_tensor.to(device)
            src_mask, _ = create_padding_mask(src, src, pad_idx=0)

            # 解码生成翻译
            if decode_method == 'greedy':
                output_indices = greedy_decode(
                    model, src, src_mask,
                    max_len=config['inference']['max_len'],
                    start_token=tgt_vocab['<sos>'],
                    end_token=tgt_vocab['<eos>'],
                    device=device
                )
            elif decode_method == 'beam_search':
                output_indices = beam_search_decode(
                    model, src, src_mask,
                    max_len=config['inference']['max_len'],
                    start_token=tgt_vocab['<sos>'],
                    end_token=tgt_vocab['<eos>'],
                    beam_width=beam_width,
                    device=device
                )
            else:
                raise ValueError(f"Unknown decode method: {decode_method}")

            # 将索引转换为单词
            pred_indices = output_indices[0].tolist()
            pred_words = []
            for idx in pred_indices:
                if idx == tgt_vocab['<eos>']:
                    break
                if idx not in [tgt_vocab['<pad>'], tgt_vocab['<sos>']]:
                    pred_words.append(idx_to_word_tgt.get(idx, '<unk>'))

            translation = ' '.join(pred_words)

            # 保存结果
            if dataset.has_reference:
                results.append({
                    'source': src_sentence,
                    'reference': ref_sentence,
                    'translation': translation
                })
            else:
                results.append({
                    'source': src_sentence,
                    'translation': translation
                })

    # 保存结果到TSV文件
    output_path = os.path.join(output_dir, config['inference']['output_file'])

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 保存为TSV文件
    if dataset.has_reference:
        results_df.to_csv(output_path, sep='\t', index=False,
                          columns=['source', 'reference', 'translation'])
    else:
        results_df.to_csv(output_path, sep='\t', index=False,
                          columns=['source', 'translation'])

    # 打印统计信息
    elapsed_time = time.time() - start_time
    avg_time_per_sentence = elapsed_time / len(dataset)

    print(f"\nInference completed!")
    print(f"Total sentences: {len(dataset)}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per sentence: {avg_time_per_sentence:.3f} seconds")
    print(f"Results saved to: {output_path}")

    # 显示前几个结果示例
    print("\nSample translations:")
    print("-" * 80)
    for i in range(min(5, len(results))):
        print(f"Source: {results[i]['source'][:50]}...")
        if 'reference' in results[i]:
            print(f"Reference: {results[i]['reference'][:50]}...")
        print(f"Translation: {results[i]['translation'][:50]}...")
        print("-" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with trained Transformer model")
    parser.add_argument("--config", type=str, default="options/inference_config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.config)