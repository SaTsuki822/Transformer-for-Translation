import os
import random
import pandas as pd
from typing import List, Tuple


def split_translation_dataset():
    """
    将TSV格式的中英翻译数据集分割为训练集、验证集和测试集

    流程：
    1. 先抽出20条作为测试集
    2. 剩余数据按8:2分割为训练集和验证集
    """

    # ========== 配置参数（请根据需要修改） ==========

    # 源文件路径（TSV格式，格式为：英文\t中文）
    SOURCE_FILE = r"F:\lyk寒假\LLM\transformer\data\news-commentary-v18.en-zh.tsv"

    # 输出文件路径
    OUTPUT_DIR = r"F:\lyk寒假\LLM\transformer\data"
    TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.tsv")
    VALID_FILE = os.path.join(OUTPUT_DIR, "valid.tsv")
    TEST_FILE = os.path.join(OUTPUT_DIR, "test.tsv")

    # 数据集分割参数
    TEST_SIZE = 20  # 测试集大小（条数）
    TRAIN_RATIO = 0.8  # 训练集比例（在剩余数据中）

    # 随机种子（保证可重复性）
    RANDOM_SEED = 42

    # ========== 数据加载 ==========

    print(f"正在读取源文件: {SOURCE_FILE}")

    try:
        # 读取TSV文件（假设没有表头）
        df = pd.read_csv(SOURCE_FILE, sep='\t', header=None, names=['en', 'zh'])
        print(f"成功读取 {len(df)} 条翻译对")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {SOURCE_FILE}")
        print("请确保源文件存在并检查路径是否正确")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 检查数据量是否足够
    if len(df) < TEST_SIZE:
        print(f"错误: 数据量 ({len(df)}) 小于测试集大小 ({TEST_SIZE})")
        return

    # ========== 数据分割 ==========

    # 设置随机种子以确保可重复性
    random.seed(RANDOM_SEED)

    # 创建所有索引的列表
    all_indices = list(range(len(df)))

    # 随机打乱索引
    random.shuffle(all_indices)

    # 1. 抽取测试集
    test_indices = all_indices[:TEST_SIZE]
    remaining_indices = all_indices[TEST_SIZE:]

    print(f"测试集抽取 {len(test_indices)} 条")

    # 2. 分割剩余数据为训练集和验证集
    split_point = int(len(remaining_indices) * TRAIN_RATIO)
    train_indices = remaining_indices[:split_point]
    valid_indices = remaining_indices[split_point:]

    print(f"训练集: {len(train_indices)} 条")
    print(f"验证集: {len(valid_indices)} 条")

    # ========== 创建数据子集 ==========

    train_df = df.iloc[train_indices].reset_index(drop=True)
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    # ========== 保存分割后的数据集 ==========

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 保存为TSV文件（无表头）
        train_df.to_csv(TRAIN_FILE, sep='\t', index=False, header=False)
        valid_df.to_csv(VALID_FILE, sep='\t', index=False, header=False)
        test_df.to_csv(TEST_FILE, sep='\t', index=False, header=False)

        print(f"\n数据集已成功分割并保存到目录: {OUTPUT_DIR}")
        print(f"训练集: {TRAIN_FILE} ({len(train_df)} 条)")
        print(f"验证集: {VALID_FILE} ({len(valid_df)} 条)")
        print(f"测试集: {TEST_FILE} ({len(test_df)} 条)")

    except Exception as e:
        print(f"保存文件时出错: {e}")
        return

    # ========== 显示样本信息 ==========

    print("\n=== 数据集样本预览 ===")
    print("\n测试集前3条:")
    for i in range(min(3, len(test_df))):
        print(f"  英文: {test_df.iloc[i]['en'][:50]}...")
        print(f"  中文: {test_df.iloc[i]['zh'][:50]}...")
        print()

    print("\n训练集前3条:")
    for i in range(min(3, len(train_df))):
        print(f"  英文: {train_df.iloc[i]['en'][:50]}...")
        print(f"  中文: {train_df.iloc[i]['zh'][:50]}...")
        print()

    # ========== 数据统计信息 ==========

    print("\n=== 数据集统计信息 ===")
    print(f"总数据量: {len(df)} 条")
    print(f"训练集占比: {len(train_df) / len(df) * 100:.1f}%")
    print(f"验证集占比: {len(valid_df) / len(df) * 100:.1f}%")
    print(f"测试集占比: {len(test_df) / len(df) * 100:.1f}%")

    # 统计中英文长度信息
    def compute_length_stats(dataframe, label):
        en_lengths = dataframe['en'].str.split().str.len()
        zh_lengths = dataframe['zh'].str.split().str.len()

        print(f"\n{label}文本长度统计:")
        print(f"  英文平均长度: {en_lengths.mean():.1f} 词")
        print(f"  中文平均长度: {zh_lengths.mean():.1f} 词")
        print(f"  英文最长: {en_lengths.max()} 词")
        print(f"  中文最长: {zh_lengths.max()} 词")

    compute_length_stats(train_df, "训练集")
    compute_length_stats(valid_df, "验证集")
    compute_length_stats(test_df, "测试集")


def create_sample_dataset():
    """
    创建一个示例数据集（如果原文件不存在）
    用于测试脚本功能
    """
    sample_data = []

    # 生成一些示例翻译对
    translations = [
        ("Hello, how are you?", "你好，最近怎么样？"),
        ("I love programming.", "我喜欢编程。"),
        ("The weather is nice today.", "今天天气很好。"),
        ("Can you help me?", "你能帮我吗？"),
        ("Where is the nearest station?", "最近的车站在哪里？"),
        ("I need to buy some groceries.", "我需要买些日用品。"),
        ("What time is the meeting?", "会议什么时候开始？"),
        ("This is a great book.", "这是一本很棒的书。"),
        ("I don't understand.", "我不明白。"),
        ("Could you repeat that?", "你能重复一遍吗？"),
        ("The food was delicious.", "食物很美味。"),
        ("I'm learning Chinese.", "我正在学习中文。"),
        ("What's your name?", "你叫什么名字？"),
        ("How old are you?", "你多大了？"),
        ("Where are you from?", "你来自哪里？"),
        ("Nice to meet you.", "很高兴认识你。"),
        ("Thank you very much.", "非常感谢。"),
        ("You're welcome.", "不客气。"),
        ("I'm sorry.", "对不起。"),
        ("Good morning!", "早上好！"),
        ("Good night!", "晚安！"),
        ("See you tomorrow.", "明天见。"),
        ("What do you do?", "你是做什么工作的？"),
        ("I'm a student.", "我是一名学生。"),
        ("Do you speak English?", "你会说英语吗？"),
        ("Yes, a little.", "是的，会一点。"),
        ("How much does it cost?", "这个多少钱？"),
        ("Where is the bathroom?", "洗手间在哪里？"),
        ("I'm lost.", "我迷路了。"),
        ("Call the police!", "报警！"),
    ]

    # 添加更多翻译对，使总数达到100
    for i in range(70):  # 已经添加了30条
        en = f"This is sample English sentence {i + 31}."
        zh = f"这是示例中文句子 {i + 31}。"
        translations.append((en, zh))

    # 保存为TSV文件
    with open("translation_dataset.tsv", "w", encoding="utf-8") as f:
        for en, zh in translations:
            f.write(f"{en}\t{zh}\n")

    print("已创建示例数据集: translation_dataset.tsv")
    print(f"包含 {len(translations)} 条翻译对")


if __name__ == "__main__":
    # 检查源文件是否存在，如果不存在则创建示例数据集
    if not os.path.exists("translation_dataset.tsv"):
        print("未找到源文件，正在创建示例数据集...")
        create_sample_dataset()
        print()

    # 执行数据集分割
    split_translation_dataset()

    print("\n=== 脚本执行完成 ===")
    print("\n使用说明:")
    print("1. 将您的中英翻译数据集保存为 'translation_dataset.tsv'")
    print("2. 每行格式: 英文句子\\t中文句子")
    print("3. 确保文件使用UTF-8编码")
    print("4. 重新运行此脚本即可分割数据集")