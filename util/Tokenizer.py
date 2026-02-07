# 分词工具类
import re
import jieba

class Tokenizer:
    """简单的分词工具，支持英文和中文"""

    @staticmethod
    def tokenize_en(text):
        """英文分词：按空格分割，保留标点符号"""
        # 在标点符号前后添加空格以便分割
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        # 将连续多个空格替换为单个空格
        text = re.sub(r'\s{2,}', ' ', text)
        # 转换为小写并分割
        tokens = text.strip().lower().split()
        return tokens

    @staticmethod
    def tokenize_zh(text):
        """中文分词：使用jieba分词"""
        # 使用jieba进行分词
        tokens = list(jieba.cut(text.strip()))
        # 过滤空字符
        tokens = [token for token in tokens if token.strip()]
        return tokens

    @staticmethod
    def tokenize(text, lang='en'):
        """根据语言选择分词方法"""
        if lang == 'en':
            return Tokenizer.tokenize_en(text)
        elif lang == 'zh':
            return Tokenizer.tokenize_zh(text)
        else:
            # 默认按空格分割
            return text.strip().split()