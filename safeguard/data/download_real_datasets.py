import os
from safeguard.paths import DATASETS_DIR
import requests
import pandas as pd
import json
import time
from urllib.parse import urlparse

# 多个真实数据集源
DATASET_SOURCES = [
    {
        'name': 'Cresci-2017 Twitter Bot Dataset',
        'url': 'https://raw.githubusercontent.com/mkearney/tweetbotornot_paper/master/data/data.rds',
        'format': 'rds',
        'description': '学术研究数据集，包含大量Twitter机器人数据'
    },
    {
        'name': 'Botometer Feedback Dataset',
        'url': 'https://botometer.osome.iu.edu/bot-repository/datasets/cresci-2017/cresci-2017.csv',
        'format': 'csv',
        'description': '来自印第安纳大学的机器人检测数据集'
    },
    {
        'name': 'Kaggle Twitter Bot Detection',
        'url': 'https://www.kaggle.com/api/v1/datasets/download/volodymyrgavrysh/twitter-bot-or-not-dataset',
        'format': 'zip',
        'description': 'Kaggle上的Twitter机器人检测数据集'
    },
    {
        'name': 'Twitter Bot Detection Dataset',
        'url': 'https://raw.githubusercontent.com/rahulvigneswaran/Twitter-Bot-Detection/main/Twitter-Account-data.csv',
        'format': 'csv',
        'description': 'GitHub上的Twitter机器人检测数据集'
    },
    {
        'name': 'Social Media Bot Detection',
        'url': 'https://raw.githubusercontent.com/r-spacex/twitter-bot-detection-dataset/main/twitter_human_bots_dataset.csv',
        'format': 'csv',
        'description': '社交媒体机器人检测数据集'
    }
]

def download_dataset(source, timeout=30):
    """尝试下载数据集"""
    print(f"\n正在尝试下载: {source['name']}")
    print(f"URL: {source['url']}")
    print(f"描述: {source['description']}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(source['url'], headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # 生成文件名
        parsed_url = urlparse(source['url'])
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = f"{source['name'].lower().replace(' ', '_')}.{source['format']}"
        
        filepath = os.path.join(DATASETS_DIR, filename)
        
        # 保存文件
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ 下载成功: {filepath}")
        print(f"文件大小: {os.path.getsize(filepath)} 字节")
        
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return None

def process_csv_dataset(filepath):
    """处理CSV格式的数据集"""
    try:
        df = pd.read_csv(filepath)
        print(f"CSV列名: {list(df.columns)}")
        print(f"数据形状: {df.shape}")
        print(f"前5行数据:")
        print(df.head())
        
        # 尝试识别关键列
        username_col = None
        bio_col = None
        label_col = None
        
        # 查找用户名列
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['username', 'screen_name', 'user', 'id', 'name']):
                username_col = col
                break
        
        # 查找简介列
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['description', 'bio', 'text', 'content']):
                bio_col = col
                break
        
        # 查找标签列
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['bot', 'label', 'class', 'type']):
                label_col = col
                break
        
        print(f"\n识别的列:")
        print(f"用户名列: {username_col}")
        print(f"简介列: {bio_col}")
        print(f"标签列: {label_col}")
        
        if username_col and label_col:
            return True, df, username_col, bio_col, label_col
        else:
            return False, df, None, None, None
            
    except Exception as e:
        print(f"❌ 处理CSV失败: {e}")
        return False, None, None, None, None

def convert_to_our_format(df, username_col, bio_col, label_col):
    """转换为我们的格式"""
    samples = []
    labels = []
    
    for _, row in df.iterrows():
        sample = {
            'username': str(row.get(username_col, '')),
            'email': '',
            'bio': str(row.get(bio_col, '')) if bio_col else '',
            'time_to_fill': '',
            'typing_pattern': '',
            'navigation': ''
        }
        samples.append(sample)
        
        # 处理标签
        label_value = row.get(label_col, 0)
        if isinstance(label_value, str):
            # 如果是字符串，转换为数字
            label_value = 1 if 'bot' in label_value.lower() else 0
        else:
            # 如果是数字，确保是0或1
            label_value = 1 if label_value == 1 else 0
            
        labels.append(label_value)
    
    return samples, labels

def main():
    """主函数"""
    print("🔍 开始寻找真实的人机检测数据集...")
    print("=" * 60)
    
    # 创建datasets目录
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    successful_datasets = []
    
    # 尝试下载每个数据集
    for source in DATASET_SOURCES:
        filepath = download_dataset(source)
        if filepath:
            if source['format'] == 'csv':
                success, df, username_col, bio_col, label_col = process_csv_dataset(filepath)
                if success:
                    samples, labels = convert_to_our_format(df, username_col, bio_col, label_col)
                    
                    # 保存为我们的格式
                    output_file = str(DATASETS_DIR / f"real_dataset_{len(successful_datasets)}.json")
                    output = {
                        'samples': samples,
                        'labels': labels,
                        'source': source['name']
                    }
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output, f, ensure_ascii=False, indent=2)
                    
                    print(f"✅ 成功处理并保存到: {output_file}")
                    print(f"   样本数量: {len(samples)}")
                    print(f"   机器人数量: {sum(labels)}")
                    print(f"   人类数量: {len(labels) - sum(labels)}")
                    
                    successful_datasets.append({
                        'name': source['name'],
                        'file': output_file,
                        'samples': len(samples),
                        'bots': sum(labels),
                        'humans': len(labels) - sum(labels)
                    })
                else:
                    print("❌ 无法识别必要的列")
            else:
                print(f"⚠️  暂不支持 {source['format']} 格式")
        
        # 等待一下再尝试下一个
        time.sleep(2)
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 下载总结:")
    if successful_datasets:
        for i, dataset in enumerate(successful_datasets):
            print(f"{i+1}. {dataset['name']}")
            print(f"   文件: {dataset['file']}")
            print(f"   样本: {dataset['samples']} (机器人: {dataset['bots']}, 人类: {dataset['humans']})")
        
        # 选择最大的数据集作为主要数据集
        best_dataset = max(successful_datasets, key=lambda x: x['samples'])
        print(f"\n🏆 推荐使用: {best_dataset['name']}")
        print(f"   文件: {best_dataset['file']}")
        
        # 复制为默认文件名
        import shutil
        shutil.copy(best_dataset['file'], str(DATASETS_DIR / 'real_dataset.json'))
        print(f"✅ 已复制为默认文件: data/datasets/real_dataset.json")
        
    else:
        print("❌ 没有成功下载任何数据集")
        print("建议:")
        print("1. 检查网络连接")
        print("2. 尝试手动下载数据集")
        print("3. 使用我们提供的本地示例数据")

if __name__ == "__main__":
    main() 