import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# 自定义的简易分词（你可以换成自己的逻辑）
def custom_word_tokenize(text):
    """简单的分词函数，不依赖NLTK资源"""
    # 首先用标点符号将文本分割
    text = re.sub(r'([.,!?;:"])', r' \1 ', text)
    # 然后分割空白字符
    words = [word.strip() for word in text.split() if word.strip()]
    return words

def custom_sent_tokenize(text):
    """简单的句子切分函数，不依赖NLTK资源"""
    # 使用常见的句子结束标记进行分割
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def analyze_platform_user(user_data):
    bio = user_data.get("bio", "")
    features = {}

    # ========== 自定义分词 ==========
    words_custom = custom_word_tokenize(bio)
    sentences_custom = custom_sent_tokenize(bio)
    features["custom_word_count"] = len(words_custom)
    features["custom_avg_word_length"] = np.mean([len(w) for w in words_custom]) if words_custom else 0
    features["custom_sentence_count"] = len(sentences_custom)

    # ========== NLTK 分词 ==========
    try:
        words_nltk = word_tokenize(bio)
    except LookupError:
        nltk.download("punkt")
        words_nltk = word_tokenize(bio)

    try:
        sentences_nltk = sent_tokenize(bio)
    except LookupError:
        nltk.download("punkt")
        sentences_nltk = sent_tokenize(bio)

    features["nltk_word_count"] = len(words_nltk)
    features["nltk_avg_word_length"] = np.mean([len(w) for w in words_nltk]) if words_nltk else 0
    features["nltk_sentence_count"] = len(sentences_nltk)

    return features


def main():
    test_user = {
        "bio": "I love machine learning. It's amazing!"
    }

    result = analyze_platform_user(test_user)
    print(result)


if __name__ == "__main__":
    main()
