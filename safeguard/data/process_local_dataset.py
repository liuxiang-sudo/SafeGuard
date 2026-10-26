import os
from safeguard.paths import DATASETS_DIR
import pandas as pd
import json

# 使用本地数据集文件
DATASET_CSV = DATASETS_DIR / 'twitter_real_data.csv'
OUTPUT_JSON = DATASETS_DIR / 'real_dataset.json'

def main():
    if not os.path.exists(DATASET_CSV):
        print(f"错误: 本地数据集文件 {DATASET_CSV} 不存在")
        print("请确保数据集文件已创建")
        exit()

    print(f'使用本地数据集: {DATASET_CSV}')

    # 2. 读取并处理
    print('正在处理数据集...')
    try:
        df = pd.read_csv(DATASET_CSV)
        print(f"成功读取 {len(df)} 条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        exit()

    samples = []
    labels = []

    for _, row in df.iterrows():
        sample = {
            'username': str(row.get('username', '')),
            'email': '',
            'bio': str(row.get('description', '')),
            'time_to_fill': '',
            'typing_pattern': '',
            'navigation': ''
        }
        samples.append(sample)
        # bot 列包含 0(人类) 或 1(机器人)
        bot_label = int(row.get('bot', 0))
        labels.append(bot_label)

    # 3. 保存为JSON
    output = {
        'samples': samples,
        'labels': labels
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'已保存处理好的数据到 {OUTPUT_JSON}')
    print(f'包含 {len(samples)} 个样本，其中 {sum(labels)} 个机器人，{len(labels) - sum(labels)} 个人类') 

if __name__ == "__main__":
    main()
