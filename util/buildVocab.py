# 构建词汇表
import os
from collections import Counter
import pandas as pd

from Tokenizer import Tokenizer

def build_vocab(data_paths, lang='en', min_freq=2, max_vocab_size=None):
    """
    从数据集中构建词汇表
    data_paths: 数据集路径列表
    lang: 语言类型 ('en' 或 'zh')
    min_freq: 最小词频
    max_vocab_size: 词汇表最大大小（可选）
    """
    word_freq = Counter()

    # 统计词频
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} does not exist")
            continue

        try:
            df = pd.read_csv(data_path, sep='\t', header=None, names=['src', 'tgt'])

            # 选择要处理的列
            if lang == 'en':
                sentences = df['src']
            elif lang == 'zh':
                sentences = df['tgt']
            else:
                print(f"Warning: Unknown language {lang}, using source column")
                sentences = df['src']

            for sentence in sentences:
                sentence = str(sentence)
                if not sentence.strip():  # 跳过空句子
                    continue

                # 分词
                tokens = Tokenizer.tokenize(sentence, lang)
                word_freq.update(tokens)

        except Exception as e:
            print(f"Error processing {data_path}: {e}")
            continue

    print(f"Total unique words in {lang}: {len(word_freq)}")
    print(f"Most common 10 words: {word_freq.most_common(10)}")

    # 构建词汇表
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }

    # 添加高频词
    idx = 4
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    for word, freq in sorted_words:
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

            # 如果设置了最大词汇表大小，则限制词汇表大小
            if max_vocab_size and len(vocab) >= max_vocab_size:
                print(f"Reached max vocabulary size {max_vocab_size} for {lang}")
                break

    print(f"{lang} vocabulary size: {len(vocab)}")
    print(
        f"Vocabulary coverage: {sum(freq for word, freq in word_freq.most_common(len(vocab) - 4)) / sum(word_freq.values()) * 100:.2f}%")

    return vocab


# 保存和加载词汇表
def save_vocab(vocab, filepath):
    """保存词汇表到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{word}\t{idx}\n")
    print(f"Saved vocabulary to {filepath}")

def load_vocab(filepath):
    """从文件加载词汇表"""
    vocab = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            vocab[word] = int(idx)
    print(f"Loaded vocabulary from {filepath}, size: {len(vocab)}")
    return vocab

# 指定数据集
data_path = "/tmp/pycharm_project_647/data/news-commentary-v18.en-zh.tsv"
vocab_dir = 'vocab'
os.makedirs(vocab_dir, exist_ok=True)

src_vocab_path = os.path.join(vocab_dir, f"en_vocab.txt")
tgt_vocab_path = os.path.join(vocab_dir, f"zh_vocab.txt")

print("Building new vocabulary...")
src_vocab = build_vocab(
    data_paths=[data_path],
    lang='en',
    min_freq=1,
    max_vocab_size=200000
)
tgt_vocab = build_vocab(
    data_paths=[data_path],
    lang='zh',
    min_freq=1,
    max_vocab_size=200000
)

# 保存词汇表
save_vocab(src_vocab, src_vocab_path)
save_vocab(tgt_vocab, tgt_vocab_path)