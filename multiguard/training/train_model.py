"""Shared data loading and training entry points for CLI and Web uploads."""
import argparse
import csv
import json
import os
from pathlib import Path
import random

from multiguard.data.validation import validate_training_data
from multiguard.detection.ml_detector import ml_detector
from multiguard.paths import TRAINING_SAMPLES_DIR, PROCESSED_DIR, ensure_runtime_dirs

def generate_synthetic_data(num_samples=1000):
    """
    生成用于训练的合成数据。
    
    Args:
        num_samples (int): 要生成的样本数量
        
    Returns:
        tuple: (data, labels) 数据和标签列表
    """
    if not isinstance(num_samples, int) or num_samples < 2:
        raise ValueError('合成样本数量必须至少为 2')
    print(f"生成 {num_samples} 个合成样本...")
    
    data = []
    labels = []
    
    # 人类用户名特征
    human_first_names = ["张", "王", "李", "赵", "刘", "陈", "杨", "黄", "周", "吴", "郑", "冯", "朱", "秦", "苏", "潘", "田", "John", "Mary", "David", "Sarah", "Michael", "Emma", "Daniel", "Olivia"]
    human_last_names = ["伟", "芳", "娜", "敏", "静", "强", "杰", "磊", "洋", "艳", "勇", "军", "霞", "明", "燕", "超", "刚", "平", "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis"]
    
    # 生成人类样本(50%)
    for i in range(num_samples // 2):
        username = random.choice(human_first_names) + random.choice(human_last_names)
        if random.random() < 0.2:  # 有时添加数字
            username += str(random.randint(0, 99))
        
        email_domains = ["gmail.com", "outlook.com", "yahoo.com", "hotmail.com", "qq.com", "163.com", "126.com"]
        email = username.lower() + "@" + random.choice(email_domains)
        
        # 生成合理的人类自我介绍
        bios = [
            f"你好，我是{username}，喜欢{random.choice(['读书', '旅行', '音乐', '电影', '摄影', '运动', '美食'])}和{random.choice(['游泳', '跑步', '骑行', '爬山', '画画', '钢琴'])}。",
            f"大家好，我叫{username}，是一名{random.choice(['学生', '老师', '工程师', '医生', '律师', '设计师', '摄影师', '作家'])}。",
            f"欢迎来到我的主页，我是{username}，热爱生活，喜欢分享。",
            f"Hi, I'm {username}. I enjoy {random.choice(['reading', 'traveling', 'music', 'movies', 'photography', 'sports', 'cooking'])} and {random.choice(['swimming', 'running', 'cycling', 'hiking', 'painting', 'playing piano'])}.",
            f"Hello everyone, I'm {username}, a {random.choice(['student', 'teacher', 'engineer', 'doctor', 'lawyer', 'designer', 'photographer', 'writer'])}."
        ]
        bio = random.choice(bios)
        
        # 生成真实的用户行为模式
        time_to_fill = random.uniform(30.0, 300.0)  # 30秒到5分钟
        
        # 生成真实的打字模式（间隔在100-300ms之间的随机值）
        typing_intervals = [str(random.uniform(100, 300)) for _ in range(random.randint(10, 50))]
        typing_pattern = ",".join(typing_intervals)
        
        # 创建样本数据
        sample = {
            "username": username,
            "email": email,
            "bio": bio,
            "register_date": "",
            "ip_address": f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "time_to_fill": str(time_to_fill),
            "typing_pattern": typing_pattern,
            "navigation": str(random.randint(0, 3)),
            "screen_size": f"{random.choice([1366, 1440, 1536, 1920, 2560])}x{random.choice([768, 900, 1080, 1440])}",
            "user_agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(70, 100)}.0.{random.randint(1000, 5000)}.{random.randint(100, 999)}",
            "referrer": random.choice(["https://www.google.com/", "https://www.baidu.com/", "https://www.bing.com/", "https://www.yahoo.com/", ""]),
            "platform": "general"
        }
        
        data.append(sample)
        labels.append(0)  # 0表示人类
    
    # 生成机器人样本(50%)
    for i in range(num_samples - len(data)):
        # 生成机器人用户名 - 通常有更多的数字和特殊字符
        if random.random() < 0.7:  # 70%的机会生成带有大量数字或特殊字符的用户名
            username = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789_") for _ in range(random.randint(8, 15)))
            username += random.choice(["bot", "auto", "ai", "robot", "proxy", "official", "service", "system", ""])
            if random.random() < 0.3:
                username += str(random.randint(1000, 9999))
        else:
            # 剩下的使用更自然的名字，但通常带有数字
            username = random.choice(human_first_names) + random.choice(human_last_names) + str(random.randint(100, 9999))
        
        # 生成邮箱 - 更可能使用不太常见的域名
        email_domains = ["gmail.com", "qq.com", "163.com", "protonmail.com", "mail.ru", "yandex.ru", "temp-mail.org"]
        email = username.lower() + "@" + random.choice(email_domains)
        
        # 生成机器人的自我介绍，通常更简短、机械或充满关键词
        bot_bios = [
            f"Official bot account for {username} services.",
            f"Hi, I'm {username}! Click my profile to learn more or visit: http://bit.ly/2rnd0m",
            f"Automated updates from {username} system",
            f"关注我，获取最新{random.choice(['优惠', '折扣', '资讯', '消息'])}",
            f"Follow for daily {random.choice(['tips', 'updates', 'news', 'promotions'])}",
            f"Promoting the best {random.choice(['products', 'services', 'deals', 'offers'])} daily",
            "",  # 空生物
            f"Welcome to my page! Check out my profile for more information."
        ]
        bio = random.choice(bot_bios)
        
        # 机器人通常填表很快
        time_to_fill = random.uniform(2.0, 20.0)  # 2-20秒
        
        # 机器人的打字模式通常很规律或极快
        if random.random() < 0.5:
            # 非常规律的打字模式
            base_interval = random.uniform(50, 100)
            typing_intervals = [str(base_interval + random.uniform(-5, 5)) for _ in range(random.randint(5, 15))]
        else:
            # 极快的打字速度
            typing_intervals = [str(random.uniform(10, 50)) for _ in range(random.randint(5, 15))]
        
        typing_pattern = ",".join(typing_intervals)
        
        # 创建样本数据
        sample = {
            "username": username,
            "email": email,
            "bio": bio,
            "register_date": "",
            "ip_address": f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            "time_to_fill": str(time_to_fill),
            "typing_pattern": typing_pattern,
            "navigation": "0",  # 机器人通常没有导航历史
            "screen_size": f"{random.choice([1366, 1440, 1536, 1920, 2560])}x{random.choice([768, 900, 1080, 1440])}",
            "user_agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(70, 100)}.0.{random.randint(1000, 5000)}.{random.randint(100, 999)}",
            "referrer": random.choice(["", "https://www.google.com/", "https://www.baidu.com/"]),
            "platform": "general"
        }
        
        data.append(sample)
        labels.append(1)  # 1表示机器人
    
    # 打乱数据
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data, labels = zip(*combined)
    
    return list(data), list(labels)

def load_data_from_file(filepath):
    """Read upload JSON, legacy record JSON, or labeled CSV consistently.

    Invalid data raises ValueError; callers must not silently train on other data.
    """
    path = Path(filepath)
    if path.suffix.lower() == '.json':
        with path.open(encoding='utf-8-sig') as stream:
            dataset = json.load(stream)
        if isinstance(dataset, dict):
            samples = dataset.get('samples', dataset.get('data'))
            labels = dataset.get('labels')
        elif isinstance(dataset, list):
            if not all(isinstance(item, dict) and 'data' in item and 'is_bot' in item
                       for item in dataset):
                raise ValueError('记录列表必须包含 data 和 is_bot 字段')
            samples = [item['data'] for item in dataset]
            labels = [item['is_bot'] for item in dataset]
        else:
            raise ValueError('JSON 必须为样本对象或记录列表')
    elif path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'username', 'email', 'label'}.issubset(reader.fieldnames or []):
                raise ValueError('CSV 必须包含 username、email、label 列')
            samples, labels = [], []
            for row in reader:
                labels.append(row.pop('label'))
                samples.append(row)
    else:
        raise ValueError('仅支持 JSON 或 CSV 数据集')
    return validate_training_data(samples, labels, require_both_classes=False)


def combine_with_default_dataset(user_data, user_labels, default_size=500):
    """
    Combine user dataset with synthetic default dataset
    
    Args:
        user_data (list): User provided data samples
        user_labels (list): User provided labels
        default_size (int): Number of synthetic samples to generate
        
    Returns:
        list: Combined data
        list: Combined labels
    """
    # 生成合成数据
    default_data, default_labels = generate_synthetic_data(default_size)
    
    # 合并数据集
    combined_data = user_data + default_data
    combined_labels = user_labels + default_labels
    
    return combined_data, combined_labels

def train_model_from_file(filepath, mix_with_default=False, default_size=500, detector=None):
    """
    Train model from a user uploaded file
    
    Args:
        filepath (str): Path to the data file
        mix_with_default (bool): Whether to mix with synthetic default data
        default_size (int): Number of synthetic samples to generate if mixing
        
    Returns:
        float: Training accuracy
    """
    # 加载用户数据
    data, labels = load_data_from_file(filepath)
    
    if data is None or len(data) == 0:
        raise ValueError("无法从文件加载有效数据")
    
    # 如果需要与默认数据集混合
    if mix_with_default:
        data, labels = combine_with_default_dataset(data, labels, default_size)
    
    # 训练模型
    active_detector = detector if detector is not None else ml_detector
    accuracy = active_detector.train(data, labels)
    
    print(f"模型训练完成，使用了 {len(data)} 个样本。")
    print(f"训练准确率: {accuracy:.2%}")
    
    # 保存训练数据样例以供参考
    sample_dir = str(TRAINING_SAMPLES_DIR)
    os.makedirs(sample_dir, exist_ok=True)
    sample_filename = os.path.join(sample_dir, f'sample_{os.path.basename(filepath)}')
    
    with open(sample_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'samples': data[:5],
            'labels': labels[:5]
        }, f, indent=2)
    
    print(f"将训练数据样例保存到 '{sample_filename}'")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='训练机器人检测 ML 模型')
    parser.add_argument('--data', type=str, help='标记数据文件的路径（JSON 或 CSV）')
    parser.add_argument('--synth', type=int, default=1000, help='未提供数据时的合成样本数量')
    args = parser.parse_args()
    try:
        if args.data:
            data, labels = load_data_from_file(args.data)
        else:
            data, labels = generate_synthetic_data(args.synth)
        data, labels = validate_training_data(data, labels)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    ensure_runtime_dirs()
    accuracy = ml_detector.train(data, labels)
    print(f"模型训练完成，使用了 {len(data)} 个样本。")
    print(f"训练集准确率: {accuracy:.2%}（不代表独立测试集表现）")
    output = PROCESSED_DIR / 'real_dataset.json'
    with output.open('w', encoding='utf-8') as stream:
        json.dump({'samples': data, 'labels': labels}, stream, indent=2, ensure_ascii=False)
    print(f"训练数据已保存到 {output}")


if __name__ == '__main__':
    main()
