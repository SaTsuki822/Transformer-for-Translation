# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import yaml
import time
from tqdm import tqdm
import datetime

# 导入自定义模块
try:
    from transformer.model import Transformer
    from transformer.create_mask import create_padding_mask
    from util.Tokenizer import Tokenizer
except ImportError:
    print("Warning: Some custom modules not found. Make sure they are in your Python path.")


# 设置随机种子以保证可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 数据集类
class TranslationDataset(Dataset):
    def __init__(self, data_path, src_vocab, tgt_vocab, src_lang='en', tgt_lang='zh', max_len=100):
        """
        读取 TSV 格式的翻译数据集
        格式：英文句子\t中文句子
        """
        self.data = pd.read_csv(data_path, sep='\t', header=None, names=['src', 'tgt'])
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        self.src_pad_idx = src_vocab.get('<pad>', 0)
        self.tgt_pad_idx = tgt_vocab.get('<pad>', 0)

        # 统计信息
        print(f"Loaded {len(self.data)} samples from {data_path}")
        print(f"Source language: {src_lang}, Target language: {tgt_lang}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sentence = str(self.data.iloc[idx]['src'])
        tgt_sentence = str(self.data.iloc[idx]['tgt'])

        # 将句子转换为索引序列
        src_indices = self.sentence_to_indices(src_sentence, self.src_vocab, self.src_lang)
        tgt_indices = self.sentence_to_indices(tgt_sentence, self.tgt_vocab, self.tgt_lang)

        # 添加特殊标记：<sos> 和 <eos>
        src_indices = [self.src_vocab['<sos>']] + src_indices + [self.src_vocab['<eos>']]
        tgt_indices = [self.tgt_vocab['<sos>']] + tgt_indices + [self.tgt_vocab['<eos>']]

        # 截断到最大长度-2（为sos和eos预留位置）
        max_content_len = self.max_len - 2
        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len]

        # 填充序列
        src_indices = self.pad_sequence(src_indices, self.max_len, self.src_pad_idx)
        tgt_indices = self.pad_sequence(tgt_indices, self.max_len, self.tgt_pad_idx)

        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long)
        }

    def sentence_to_indices(self, sentence, vocab, lang):
        """将句子转换为索引序列"""
        # 分词
        tokens = Tokenizer.tokenize(sentence, lang)

        indices = []
        for token in tokens:
            if token in vocab:
                indices.append(vocab[token])
            else:
                # 如果词汇表中没有该词，使用<unk>
                indices.append(vocab.get('<unk>', 1))
        return indices

    def pad_sequence(self, sequence, max_len, pad_idx):
        """填充序列到指定长度"""
        if len(sequence) < max_len:
            sequence = sequence + [pad_idx] * (max_len - len(sequence))
        return sequence[:max_len]

# 训练函数
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, log_interval):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        # print("--------tgt--------", tgt)
        # 创建掩码
        src_mask, tgt_mask = create_padding_mask(src, tgt[:, :-1], pad_idx=0)

        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)

        # 计算损失
        # 输出形状: (batch_size, tgt_seq_len-1, tgt_vocab_size)
        # 目标形状: (batch_size, tgt_seq_len-1)
        loss = criterion(
            output.contiguous().view(-1, output.size(-1)),
            tgt[:, 1:].contiguous().view(-1)
        )

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 打印日志
        if (i + 1) % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'Epoch {epoch}, Iteration {i + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s')
            total_loss = 0
            start_time = time.time()

    return total_loss / max(len(dataloader), 1)


# 验证损失计算
def evaluate_loss(model, data_loader, criterion, device):
    """评估模型在验证集上的损失"""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            # 创建掩码
            src_mask, tgt_mask = create_padding_mask(src, tgt[:, :-1], pad_idx=0)

            # 前向传播
            output = model(src, tgt[:, :-1], src_mask, tgt_mask)

            # 计算损失
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1)
            )

            total_loss += loss.item() * src.size(0)
            total_samples += src.size(0)

    model.train()
    return total_loss / total_samples if total_samples > 0 else 0

def load_vocab(filepath):
    """从文件加载词汇表"""
    vocab = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            vocab[word] = int(idx)
    print(f"Loaded vocabulary from {filepath}, size: {len(vocab)}")
    return vocab

# 主训练函数
def main(config_path):
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    set_seed(config['training']['seed'])

    # 设备设置
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建基于时间戳的保存目录
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = config['training']['save_dir']
    time_save_dir = os.path.join(base_save_dir, current_time)
    os.makedirs(time_save_dir, exist_ok=True)

    # 更新配置中的保存目录为时间戳目录
    config['training']['save_dir'] = time_save_dir

    # 保存配置文件到时间戳目录
    config_save_path = os.path.join(time_save_dir, "train_config.yaml")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # 创建损失值记录文件
    loss_file_path = os.path.join(time_save_dir, "loss_values.txt")
    with open(loss_file_path, 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,val_loss\n")

    print(f"Training session directory created: {time_save_dir}")

    # 构建或加载词汇表
    print("Building vocabulary...")
    train_path = config['data']['train_path']
    val_path = config['data']['val_path']

    # 检查是否已有保存的词汇表
    vocab_dir = config['data'].get('vocab_dir', 'vocab')


    src_vocab_path = os.path.join(vocab_dir, f"{config['data']['src_lang']}_vocab.txt")
    tgt_vocab_path = os.path.join(vocab_dir, f"{config['data']['tgt_lang']}_vocab.txt")

    print("Loading existing vocabulary...")
    src_vocab = load_vocab(src_vocab_path)
    tgt_vocab = load_vocab(tgt_vocab_path)

    # 更新词汇表大小
    config['model']['src_vocab_size'] = len(src_vocab)
    config['model']['tgt_vocab_size'] = len(tgt_vocab)

    # 创建数据集
    print("Creating datasets...")
    train_dataset = TranslationDataset(
        train_path,
        src_vocab,
        tgt_vocab,
        src_lang=config['data']['src_lang'],
        tgt_lang=config['data']['tgt_lang'],
        max_len=config['data']['max_len']
    )

    val_dataset = TranslationDataset(
        val_path,
        src_vocab,
        tgt_vocab,
        src_lang=config['data']['src_lang'],
        tgt_lang=config['data']['tgt_lang'],
        max_len=config['data']['max_len']
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 创建模型
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=config['model']['src_vocab_size'],
        tgt_vocab_size=config['model']['tgt_vocab_size'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_step_size'],
        gamma=config['training']['lr_gamma']
    )

    # 训练循环
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(1, config['training']['num_epochs'] + 1):
        # 训练一个 epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, config['training']['log_interval']
        )

        # 更新学习率
        scheduler.step()

        # 验证
        val_loss = None
        if epoch % config['training']['val_interval'] == 0:
            print(f"Evaluating on validation set...")

            # 计算验证集损失
            val_loss = evaluate_loss(
                model, val_loader, criterion, device
            )

            print(f"Validation loss: {val_loss:.4f}")

            # 更新最佳验证损失
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(
                    config['training']['save_dir'],
                    "best_model.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'src_vocab': src_vocab,
                    'tgt_vocab': tgt_vocab,
                    'config': config
                }, best_model_path)
                print(f"Saved best model to {best_model_path}")

        # 记录损失值到文件
        with open(loss_file_path, 'a', encoding='utf-8') as f:
            if val_loss is not None:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}\n")
            else:
                f.write(f"{epoch},{train_loss:.4f},\n")

        # 定期保存检查点
        if epoch % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['training']['save_dir'],
                f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss if val_loss else None,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # 保存最终模型
    final_model_path = os.path.join(
        config['training']['save_dir'],
        "transformer_final.pth"
    )
    torch.save({
        'epoch': config['training']['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'config': config
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

    # 打印训练结果摘要
    print(f"\nTraining session summary:")
    print(f"- Directory: {time_save_dir}")
    print(f"- Config saved: {config_save_path}")
    print(f"- Loss values saved: {loss_file_path}")
    print(f"- Source vocabulary size: {len(src_vocab)}")
    print(f"- Target vocabulary size: {len(tgt_vocab)}")
    print(f"- Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Transformer for translation task")
    parser.add_argument("--config", type=str, default="options/config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.config)