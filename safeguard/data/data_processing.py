import os
from safeguard.paths import DATASETS_DIR, PROCESSED_DIR, ensure_runtime_dirs
import json
import pandas as pd
import numpy as np
import argparse
import re
import random
from tqdm import tqdm
import requests
import zipfile
import io

def download_dataset(url, target_dir=DATASETS_DIR):
    """
    下载数据集到指定目录
    
    Args:
        url: 数据集URL
        target_dir: 目标目录
    """
    print(f"正在下载数据集: {url}")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # 从URL获取文件名
    filename = url.split('/')[-1]
    target_path = os.path.join(target_dir, filename)
    
    # 下载文件
    response = requests.get(url, stream=True)
    
    # 判断是否是zip文件
    if filename.endswith('.zip'):
        # 解压缩到目标目录
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"已解压缩到 {target_dir}")
    else:
        # 保存文件
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"已下载到 {target_path}")
    
    return target_dir

def process_twitter_bot_dataset(dataset_path, max_samples=5000):
    """
    处理Twitter Bot检测数据集
    
    Args:
        dataset_path: 数据集路径
        max_samples: 最大样本数
        
    Returns:
        list: 处理后的样本
        list: 对应的标签 (1为机器人, 0为人类)
    """
    print(f"正在处理Twitter Bot数据集: {dataset_path}")
    
    data = []
    labels = []
    
    try:
        # 尝试不同格式的数据集
        if os.path.isfile(dataset_path) and dataset_path.endswith('.csv'):
            # CSV格式
            df = pd.read_csv(dataset_path)
            
            # 检测相关列
            bot_label_col = None
            text_col = None
            username_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'bot' in col_lower or 'label' in col_lower or 'class' in col_lower:
                    bot_label_col = col
                elif 'text' in col_lower or 'tweet' in col_lower or 'content' in col_lower:
                    text_col = col
                elif 'user' in col_lower or 'name' in col_lower or 'screen' in col_lower:
                    username_col = col
            
            if bot_label_col and (text_col or username_col):
                # 选择样本数量
                df = df.sample(min(len(df), max_samples))
                
                for _, row in tqdm(df.iterrows(), total=len(df)):
                    # 构建样本
                    sample = {}
                    
                    # 提取用户名
                    if username_col:
                        sample['username'] = str(row[username_col])
                    else:
                        sample['username'] = f"user_{random.randint(1000, 9999)}"
                    
                    # 提取或生成邮箱
                    sample['email'] = f"{sample['username'].replace(' ', '_').lower()}@example.com"
                    
                    # 提取文本作为bio
                    if text_col and pd.notna(row[text_col]):
                        sample['bio'] = str(row[text_col])[:500]  # 限制长度
                    else:
                        sample['bio'] = "No bio available"
                    
                    # 生成其他必要字段
                    sample['ip_address'] = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
                    
                    # 根据标签生成不同的行为特征
                    is_bot = 1 if (str(row[bot_label_col]).lower() in ['1', 'true', 'bot', 'y', 'yes']) else 0
                    
                    if is_bot:
                        # Bot特征
                        sample['time_to_fill'] = str(random.uniform(1, 10))
                        sample['typing_pattern'] = ",".join([str(random.uniform(80, 120)) for _ in range(random.randint(5, 15))])
                        sample['navigation'] = "0"
                    else:
                        # 人类特征
                        sample['time_to_fill'] = str(random.uniform(20, 120))
                        typing_intervals = []
                        for _ in range(random.randint(10, 30)):
                            interval = random.normalvariate(300, 150)
                            typing_intervals.append(str(max(50, interval)))
                        sample['typing_pattern'] = ",".join(typing_intervals)
                        sample['navigation'] = str(random.randint(1, 4))
                    
                    data.append(sample)
                    labels.append(is_bot)
                
        elif os.path.isdir(dataset_path):
            # 目录格式，可能包含多个文件
            # 寻找人类和机器人数据文件
            human_files = []
            bot_files = []
            
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    file_lower = file.lower()
                    file_path = os.path.join(root, file)
                    
                    if 'human' in file_lower or 'genuine' in file_lower:
                        human_files.append(file_path)
                    elif 'bot' in file_lower or 'fake' in file_lower or 'spam' in file_lower:
                        bot_files.append(file_path)
            
            # 处理人类数据
            human_count = 0
            for file_path in human_files:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    for _, row in tqdm(df.iterrows()):
                        if human_count >= max_samples // 2:
                            break
                        
                        sample = create_sample_from_row(row, is_bot=False)
                        if sample:
                            data.append(sample)
                            labels.append(0)
                            human_count += 1
            
            # 处理机器人数据
            bot_count = 0
            for file_path in bot_files:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    for _, row in tqdm(df.iterrows()):
                        if bot_count >= max_samples // 2:
                            break
                        
                        sample = create_sample_from_row(row, is_bot=True)
                        if sample:
                            data.append(sample)
                            labels.append(1)
                            bot_count += 1
    
    except Exception as e:
        print(f"处理Twitter数据集时出错: {str(e)}")
        # 如果处理失败，返回一些合成数据
        return generate_synthetic_data(max_samples)
    
    if len(data) == 0:
        print("无法从提供的数据集生成样本，使用合成数据")
        return generate_synthetic_data(max_samples)
    
    print(f"成功处理了 {len(data)} 个样本")
    return data, labels

def create_sample_from_row(row, is_bot=False):
    """
    从一行数据创建样本
    
    Args:
        row: 数据行
        is_bot: 是否是机器人
        
    Returns:
        dict: 样本数据
    """
    sample = {}
    
    # 尝试各种可能的列名
    username = None
    text = None
    
    for col in row.index:
        col_lower = col.lower()
        val = row[col]
        
        # 跳过NaN值
        if pd.isna(val):
            continue
            
        if 'user' in col_lower or 'name' in col_lower or 'screen' in col_lower:
            username = str(val)
        elif 'text' in col_lower or 'tweet' in col_lower or 'content' in col_lower or 'description' in col_lower:
            text = str(val)
    
    if username:
        sample['username'] = username
    else:
        sample['username'] = f"user_{random.randint(1000, 9999)}"
    
    # 提取或生成邮箱
    sample['email'] = f"{sample['username'].replace(' ', '_').lower()}@example.com"
    
    # 提取文本作为bio
    if text:
        sample['bio'] = text[:500]  # 限制长度
    else:
        sample['bio'] = "No bio available"
    
    # 生成其他必要字段
    sample['ip_address'] = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
    
    # 根据标签生成不同的行为特征
    if is_bot:
        # Bot特征
        sample['time_to_fill'] = str(random.uniform(1, 10))
        sample['typing_pattern'] = ",".join([str(random.uniform(80, 120)) for _ in range(random.randint(5, 15))])
        sample['navigation'] = "0"
    else:
        # 人类特征
        sample['time_to_fill'] = str(random.uniform(20, 120))
        typing_intervals = []
        for _ in range(random.randint(10, 30)):
            interval = random.normalvariate(300, 150)
            typing_intervals.append(str(max(50, interval)))
        sample['typing_pattern'] = ",".join(typing_intervals)
        sample['navigation'] = str(random.randint(1, 4))
    
    return sample

def process_spam_dataset(dataset_path, max_samples=5000):
    """
    处理垃圾邮件数据集
    
    Args:
        dataset_path: 数据集路径
        max_samples: 最大样本数
        
    Returns:
        list: 处理后的样本
        list: 对应的标签 (1为机器人/垃圾邮件, 0为人类/正常邮件)
    """
    print(f"正在处理垃圾邮件数据集: {dataset_path}")
    
    data = []
    labels = []
    
    try:
        # 检查文件夹结构，查找spam和ham文件夹
        spam_dir = None
        ham_dir = None
        
        if os.path.isdir(dataset_path):
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    if 'spam' in item.lower():
                        spam_dir = item_path
                    elif 'ham' in item.lower() or 'legit' in item.lower():
                        ham_dir = item_path
        
        # 处理ham(正常)邮件
        if ham_dir:
            ham_files = [os.path.join(ham_dir, f) for f in os.listdir(ham_dir) if os.path.isfile(os.path.join(ham_dir, f))]
            ham_count = 0
            
            for file_path in tqdm(ham_files):
                if ham_count >= max_samples // 2:
                    break
                    
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        
                    # 提取可能的发件人信息
                    from_match = re.search(r'From:(.*?)[\r\n]', content)
                    username = "user"
                    email = "user@example.com"
                    
                    if from_match:
                        from_line = from_match.group(1).strip()
                        email_match = re.search(r'[\w\.-]+@[\w\.-]+', from_line)
                        if email_match:
                            email = email_match.group(0)
                            username = email.split('@')[0]
                    
                    # 提取主题作为bio
                    subject_match = re.search(r'Subject:(.*?)[\r\n]', content)
                    bio = "No subject"
                    
                    if subject_match:
                        bio = subject_match.group(1).strip()
                    
                    # 添加邮件正文内容
                    body_parts = content.split('\n\n', 1)
                    if len(body_parts) > 1:
                        bio += " " + body_parts[1][:300]  # 仅使用前300个字符
                    
                    # 创建样本
                    sample = {
                        'username': username,
                        'email': email,
                        'bio': bio,
                        'ip_address': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                        'time_to_fill': str(random.uniform(20, 120)),  # 人类特征
                        'navigation': str(random.randint(1, 4))
                    }
                    
                    # 人类打字模式
                    typing_intervals = []
                    for _ in range(random.randint(10, 30)):
                        interval = random.normalvariate(300, 150)
                        typing_intervals.append(str(max(50, interval)))
                    sample['typing_pattern'] = ",".join(typing_intervals)
                    
                    data.append(sample)
                    labels.append(0)  # 人类
                    ham_count += 1
                    
                except Exception as e:
                    print(f"处理邮件文件时出错 {file_path}: {str(e)}")
                    continue
        
        # 处理spam(垃圾)邮件
        if spam_dir:
            spam_files = [os.path.join(spam_dir, f) for f in os.listdir(spam_dir) if os.path.isfile(os.path.join(spam_dir, f))]
            spam_count = 0
            
            for file_path in tqdm(spam_files):
                if spam_count >= max_samples // 2:
                    break
                    
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        
                    # 提取可能的发件人信息
                    from_match = re.search(r'From:(.*?)[\r\n]', content)
                    username = f"spammer_{random.randint(1000, 9999)}"
                    email = f"{username}@spam.com"
                    
                    if from_match:
                        from_line = from_match.group(1).strip()
                        email_match = re.search(r'[\w\.-]+@[\w\.-]+', from_line)
                        if email_match:
                            email = email_match.group(0)
                            username = email.split('@')[0]
                    
                    # 提取主题作为bio
                    subject_match = re.search(r'Subject:(.*?)[\r\n]', content)
                    bio = "No subject"
                    
                    if subject_match:
                        bio = subject_match.group(1).strip()
                    
                    # 添加邮件正文内容
                    body_parts = content.split('\n\n', 1)
                    if len(body_parts) > 1:
                        bio += " " + body_parts[1][:300]  # 仅使用前300个字符
                    
                    # 创建样本
                    sample = {
                        'username': username,
                        'email': email,
                        'bio': bio,
                        'ip_address': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                        'time_to_fill': str(random.uniform(1, 10)),  # 机器人特征
                        'navigation': "0"
                    }
                    
                    # 机器人打字模式
                    sample['typing_pattern'] = ",".join([str(random.uniform(80, 120)) for _ in range(random.randint(5, 15))])
                    
                    data.append(sample)
                    labels.append(1)  # 机器人
                    spam_count += 1
                    
                except Exception as e:
                    print(f"处理邮件文件时出错 {file_path}: {str(e)}")
                    continue
    
    except Exception as e:
        print(f"处理垃圾邮件数据集时出错: {str(e)}")
        # 如果处理失败，返回一些合成数据
        return generate_synthetic_data(max_samples)
    
    if len(data) == 0:
        print("无法从提供的数据集生成样本，使用合成数据")
        return generate_synthetic_data(max_samples)
    
    print(f"成功处理了 {len(data)} 个样本")
    return data, labels

def generate_synthetic_data(num_samples=1000):
    """
    生成合成训练数据，包含机器人和人类的示例。
    
    参数:
        num_samples (int): 要生成的样本数量
        
    返回:
        list: 用户数据字典的列表
        list: 对应的标签 (1表示机器人, 0表示人类)
    """
    print(f"生成 {num_samples} 个合成样本...")
    
    data = []
    labels = []
    
    # 生成机器人示例
    for i in range(num_samples // 2):
        # 生成合成机器人数据
        bot_data = {
            'username': f"user{np.random.randint(10000, 99999)}",
            'email': f"bot{np.random.randint(1000, 9999)}@example.com",
            'bio': "I am a user of this website.",
            'ip_address': f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
            'time_to_fill': str(np.random.uniform(1, 8)),  # 表单填写非常快
            'typing_pattern': ",".join([str(np.random.uniform(80, 120)) for _ in range(np.random.randint(5, 15))]),  # 均匀打字
            'navigation': "0"  # 无导航
        }
        data.append(bot_data)
        labels.append(1)  # 1 表示机器人
    
    # 生成人类示例
    for i in range(num_samples // 2):
        # 生成更自然的人类数据
        human_names = ["john", "mary", "alex", "emma", "david", "sarah", "michael", "lisa"]
        human_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
        human_bio_templates = [
            "Hi there! I'm a {adjective} {profession} from {location}. I enjoy {hobby} in my free time.",
            "I've been working as a {profession} for {years} years. I love {hobby} and {hobby2}.",
            "Hello everyone. I'm interested in {hobby}, {hobby2}, and learning new things.",
        ]
        adjectives = ["passionate", "dedicated", "creative", "enthusiastic"]
        professions = ["developer", "designer", "teacher", "engineer", "doctor", "artist"]
        locations = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney"]
        hobbies = ["reading", "swimming", "hiking", "photography", "cooking", "traveling", "painting", "gaming"]
        
        name = np.random.choice(human_names)
        domain = np.random.choice(human_domains)
        bio_template = np.random.choice(human_bio_templates)
        bio = bio_template.format(
            adjective=np.random.choice(adjectives),
            profession=np.random.choice(professions),
            location=np.random.choice(locations),
            hobby=np.random.choice(hobbies),
            hobby2=np.random.choice(hobbies),
            years=np.random.randint(1, 15)
        )
        
        # 创建更多变化的打字模式（人类有不规则的打字）
        typing_intervals = []
        for _ in range(np.random.randint(10, 30)):
            # 为打字速度添加更多变化性
            interval = np.random.normal(300, 150)
            typing_intervals.append(str(max(50, interval)))
        
        human_data = {
            'username': name + (str(np.random.randint(1, 100)) if np.random.random() > 0.5 else ""),
            'email': f"{name}{np.random.randint(1, 100) if np.random.random() > 0.3 else ''}@{domain}",
            'bio': bio,
            'ip_address': f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
            'time_to_fill': str(np.random.uniform(20, 120)),  # 表单填写较慢
            'typing_pattern': ",".join(typing_intervals),
            'navigation': str(np.random.randint(1, 4))  # 有一些导航
        }
        data.append(human_data)
        labels.append(0)  # 0 表示人类
    
    return data, labels

def download_public_datasets():
    """
    下载一些公开的数据集用于训练
    """
    datasets_info = [
        {
            "name": "Twitter Bot Dataset (small)",
            "url": "https://raw.githubusercontent.com/mkearney/tweetbotornot_paper/master/data/data.rds",
            "type": "twitter_bot"
        },
        {
            "name": "SpamAssassin Public Corpus",
            "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
            "type": "spam"
        }
    ]
    
    downloaded_datasets = []
    
    for dataset in datasets_info:
        try:
            print(f"尝试下载数据集: {dataset['name']}")
            path = download_dataset(dataset['url'])
            downloaded_datasets.append({
                "name": dataset['name'],
                "path": path,
                "type": dataset['type']
            })
        except Exception as e:
            print(f"下载数据集失败 {dataset['name']}: {str(e)}")
    
    return downloaded_datasets

def main():
    ensure_runtime_dirs()
    parser = argparse.ArgumentParser(description='处理人机检测数据集')
    parser.add_argument('--dataset', type=str, help='数据集路径')
    parser.add_argument('--type', type=str, choices=['twitter_bot', 'spam', 'synthetic'], default='synthetic', 
                      help='数据集类型')
    parser.add_argument('--samples', type=int, default=2000, help='要处理的样本数量')
    parser.add_argument('--output', type=str, default=str(PROCESSED_DIR / 'processed_dataset.json'), help='输出文件路径')
    parser.add_argument('--download', action='store_true', help='下载公开数据集')
    
    args = parser.parse_args()
    
    if args.download:
        print("下载公开数据集...")
        downloaded_datasets = download_public_datasets()
        
        all_data = []
        all_labels = []
        
        # 处理下载的数据集
        for dataset in downloaded_datasets:
            if dataset['type'] == 'twitter_bot':
                data, labels = process_twitter_bot_dataset(dataset['path'], args.samples // len(downloaded_datasets))
            elif dataset['type'] == 'spam':
                data, labels = process_spam_dataset(dataset['path'], args.samples // len(downloaded_datasets))
            else:
                continue
                
            all_data.extend(data)
            all_labels.extend(labels)
        
        # 如果没有处理到足够的样本，添加一些合成数据
        if len(all_data) < args.samples:
            additional_samples = args.samples - len(all_data)
            print(f"添加 {additional_samples} 个合成样本以达到目标样本数")
            synth_data, synth_labels = generate_synthetic_data(additional_samples)
            all_data.extend(synth_data)
            all_labels.extend(synth_labels)
        
        # 打乱数据
        combined = list(zip(all_data, all_labels))
        random.shuffle(combined)
        all_data, all_labels = zip(*combined)
        
        # 保存处理后的数据集
        result = {
            'data': all_data,
            'labels': all_labels
        }
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"已处理 {len(all_data)} 个样本并保存到 {args.output}")
        
    elif args.dataset:
        # 处理指定的数据集
        if args.type == 'twitter_bot':
            data, labels = process_twitter_bot_dataset(args.dataset, args.samples)
        elif args.type == 'spam':
            data, labels = process_spam_dataset(args.dataset, args.samples)
        else:
            data, labels = generate_synthetic_data(args.samples)
        
        # 保存处理后的数据集
        result = {
            'data': data,
            'labels': labels
        }
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"已处理 {len(data)} 个样本并保存到 {args.output}")
    else:
        # 没有指定数据集，生成合成数据
        data, labels = generate_synthetic_data(args.samples)
        
        # 保存合成数据集
        result = {
            'data': data,
            'labels': labels
        }
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"已生成 {len(data)} 个合成样本并保存到 {args.output}")

if __name__ == "__main__":
    main()
