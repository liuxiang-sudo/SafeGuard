import json
import os
from safeguard.detection.ml_detector import MLDetector
from safeguard.paths import PROCESSED_DIR, DATASETS_DIR, EXAMPLES_DIR, ensure_runtime_dirs

def test_twibot20_integration():
    """
    测试TwiBot-20数据与现有系统的集成
    """
    print("测试TwiBot-20数据集成...")
    
    # 加载转换后的数据
    with open(PROCESSED_DIR / 'twibot20_converted_simple.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"加载了 {len(data['samples'])} 个TwiBot-20样本")
    print(f"机器人数量: {sum(data['labels'])}")
    print(f"人类数量: {len(data['labels']) - sum(data['labels'])}")
    
    # 初始化ML检测器
    detector = MLDetector()
    
    # 测试几个样本
    print("\n测试样本检测:")
    for i in range(min(5, len(data['samples']))):
        sample = data['samples'][i]
        true_label = data['labels'][i]
        
        # 使用ML检测器进行预测
        result = detector.predict_bot_probability(sample)
        
        print(f"样本 {i+1}:")
        print(f"  用户名: {sample['username']}")
        print(f"  Bio: {sample['bio'][:50]}...")
        print(f"  真实标签: {'机器人' if true_label == 1 else '人类'}")
        print(f"  预测概率: {result[0]:.2%}")
        print(f"  主要证据: {result[1][0] if result[1] else '无'}")
        print()
    
    # 计算整体准确率
    correct = 0
    total = 0
    
    for i, sample in enumerate(data['samples']):
        true_label = data['labels'][i]
        result = detector.predict_bot_probability(sample)
        predicted_label = 1 if result[0] > 0.5 else 0
        
        if predicted_label == true_label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"整体准确率: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy

def create_twitter_template():
    """
    创建Twitter平台的数据模板
    """
    template = {
        "samples": [
            {
                "platform": "twitter",
                "username": "example_twitter_user",
                "email": "user@example.com",
                "bio": "热爱分享生活点滴，关注科技和新闻",
                "register_date": "2020-01-01",
                "ip_address": "192.168.1.1",
                "time_to_fill": "45.2",
                "typing_pattern": "340,250,180,420,300",
                "navigation": "2",
                "screen_size": "1920x1080",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "referrer": "https://www.google.com/",
                "followers_count": "1000",
                "friends_count": "500",
                "statuses_count": "200",
                "verified": "false",
                "domain": "general",
                "tweet_count": "50"
            }
        ],
        "labels": [0]  # 0 表示人类, 1 表示机器人
    }
    
    with open(EXAMPLES_DIR / 'twitter_template.json', 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    print("Twitter模板已保存到: twitter_template.json")

def integrate_with_web_app():
    """
    将TwiBot-20数据集成到Web应用中
    """
    print("将TwiBot-20数据集成到Web应用...")
    
    # 复制转换后的数据到datasets目录
    if not DATASETS_DIR.exists():
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 复制数据文件
    import shutil
    shutil.copy(PROCESSED_DIR / 'enhanced_dataset_simple.json', DATASETS_DIR / 'twibot20_enhanced.json')
    
    print("数据已复制到 data/datasets/twibot20_enhanced.json")
    print("你可以在Web应用中上传这个文件进行训练")

if __name__ == "__main__":
    ensure_runtime_dirs()
    # 测试数据集成
    accuracy = test_twibot20_integration()
    
    # 创建Twitter模板
    create_twitter_template()
    
    # 集成到Web应用
    integrate_with_web_app()
    
    print("\n集成测试完成！")
    print("你可以:")
    print("1. 在Web应用中上传 'data/datasets/twibot20_enhanced.json' 进行训练")
    print("2. 使用 'twitter_template.json' 作为Twitter数据格式参考")
    print("3. 查看测试结果了解模型在TwiBot-20数据上的表现") 