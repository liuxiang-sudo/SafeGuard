import json
from multiguard.paths import TWIBOT_DIR, PROCESSED_DIR, ensure_runtime_dirs
import random
import os
from datetime import datetime

def convert_twibot20_simple():
    """
    简化版本的TwiBot-20数据转换
    """
    print("开始转换TwiBot-20数据（简化版）...")
    
    ensure_runtime_dirs()
    # 读取TwiBot-20样本数据
    with open(TWIBOT_DIR / 'TwiBot-20_sample.json', 'r', encoding='utf-8') as f:
        twibot_data = json.load(f)
    
    print(f"加载了 {len(twibot_data)} 个用户数据")
    
    # 转换为系统格式
    converted_data = []
    labels = []
    
    for user in twibot_data:
        try:
            # 提取基本信息
            user_id = str(user['ID'])
            profile = user.get('profile', {}) or {}
            tweets = user.get('tweet', []) or []
            domain = user.get('domain', 'general')
            
            # 生成用户名
            username = profile.get('screen_name', f"user_{user_id}")
            if not username or username.strip() == "":
                username = f"user_{user_id}"
            
            # 生成邮箱
            email = f"{username.lower().replace(' ', '_').replace('@', '_')}@example.com"
            
            # 提取bio（用户描述）
            bio = ""
            if profile.get('description'):
                bio = str(profile['description'])
            elif tweets and len(tweets) > 0:
                # 如果没有描述，使用第一条推文作为bio
                first_tweet = str(tweets[0])
                bio = first_tweet[:200] if len(first_tweet) > 200 else first_tweet
            
            # 如果没有bio，生成一个基于领域的
            if not bio or bio.strip() == "":
                domain_bios = {
                    'politics': '关注政治动态，分享时事观点',
                    'business': '商业分析师，关注市场趋势',
                    'entertainment': '娱乐爱好者，分享有趣内容',
                    'sports': '体育迷，热爱各种运动'
                }
                bio = domain_bios.get(domain, '热爱生活，喜欢分享')
            
            # 生成注册日期
            register_date = "2020-01-01"
            if profile.get('created_at'):
                try:
                    # 解析Twitter日期格式
                    created_date = datetime.strptime(
                        str(profile['created_at']), 
                        '%a %b %d %X %z %Y'
                    )
                    register_date = created_date.strftime('%Y-%m-%d')
                except:
                    pass
            
            # 生成IP地址
            ip_address = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
            
            # 简单的机器人判断逻辑
            is_bot = simple_bot_detection(profile, tweets, domain)
            
            # 生成行为特征
            if is_bot:
                # 机器人特征
                time_to_fill = random.uniform(2.0, 15.0)
                typing_intervals = [str(random.uniform(50, 120)) for _ in range(random.randint(5, 12))]
                navigation = "0"
            else:
                # 人类特征
                time_to_fill = random.uniform(20.0, 180.0)
                typing_intervals = [str(random.uniform(150, 400)) for _ in range(random.randint(10, 25))]
                navigation = str(random.randint(1, 3))
            
            typing_pattern = ",".join(typing_intervals)
            
            # 生成其他必要字段
            screen_size = f"{random.choice([1366, 1440, 1536, 1920, 2560])}x{random.choice([768, 900, 1080, 1440])}"
            user_agent = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(70, 100)}.0.{random.randint(1000, 5000)}.{random.randint(100, 999)}"
            referrer = random.choice(["https://www.google.com/", "https://www.baidu.com/", "https://www.bing.com/", ""])
            
            # 创建样本数据
            sample = {
                "username": username,
                "email": email,
                "bio": bio,
                "register_date": register_date,
                "ip_address": ip_address,
                "time_to_fill": str(time_to_fill),
                "typing_pattern": typing_pattern,
                "navigation": navigation,
                "screen_size": screen_size,
                "user_agent": user_agent,
                "referrer": referrer,
                "platform": "twitter",  # 标记为Twitter数据
                # 添加Twitter特有字段
                "followers_count": str(profile.get('followers_count', 0)),
                "friends_count": str(profile.get('friends_count', 0)),
                "statuses_count": str(profile.get('statuses_count', 0)),
                "verified": str(profile.get('verified', False)).lower(),
                "domain": domain,
                "tweet_count": str(len(tweets))
            }
            
            converted_data.append(sample)
            labels.append(1 if is_bot else 0)
            
        except Exception as e:
            print(f"处理用户 {user.get('ID', 'unknown')} 时出错: {e}")
            continue
    
    print(f"成功转换了 {len(converted_data)} 个样本")
    print(f"机器人数量: {sum(labels)}")
    print(f"人类数量: {len(labels) - sum(labels)}")
    
    # 保存转换后的数据
    output_data = {
        "samples": converted_data,
        "labels": labels
    }
    
    output_file = str(PROCESSED_DIR / "twibot20_converted_simple.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换后的数据已保存到: {output_file}")
    
    return output_file

def simple_bot_detection(profile, tweets, domain):
    """
    简化的机器人检测逻辑
    """
    bot_score = 0
    
    # 检查用户名
    screen_name = str(profile.get('screen_name', '')).lower()
    bot_keywords = ['bot', 'auto', 'ai', 'robot', 'official', 'service', 'system']
    if any(keyword in screen_name for keyword in bot_keywords):
        bot_score += 2
    
    # 检查描述
    description = str(profile.get('description', '')).lower()
    if any(keyword in description for keyword in bot_keywords):
        bot_score += 2
    
    # 检查推文数量
    try:
        statuses = int(profile.get('statuses_count', 0))
        if statuses > 10000:
            bot_score += 1
    except:
        pass
    
    # 检查关注者比例
    try:
        followers = int(profile.get('followers_count', 0))
        friends = int(profile.get('friends_count', 0))
        if followers > 0 and friends > 0:
            ratio = followers / friends
            if ratio > 10 or ratio < 0.1:
                bot_score += 1
    except:
        pass
    
    # 根据领域调整
    if domain == 'politics':
        bot_score += 0.5
    
    return bot_score >= 1.5

def create_enhanced_dataset_simple():
    """
    创建增强数据集（简化版）
    """
    print("创建增强数据集（简化版）...")
    
    # 转换TwiBot-20数据
    twibot_file = convert_twibot20_simple()
    
    # 加载转换后的数据
    with open(twibot_file, 'r', encoding='utf-8') as f:
        twibot_data = json.load(f)
    
    # 生成额外的合成数据以平衡数据集
    from multiguard.training.train_model import generate_synthetic_data
    
    # 生成合成数据
    synthetic_data, synthetic_labels = generate_synthetic_data(1000)
    
    # 合并数据
    combined_samples = twibot_data['samples'] + synthetic_data
    combined_labels = twibot_data['labels'] + synthetic_labels
    
    # 打乱数据
    combined = list(zip(combined_samples, combined_labels))
    random.shuffle(combined)
    combined_samples, combined_labels = zip(*combined)
    
    # 保存增强数据集
    enhanced_data = {
        "samples": list(combined_samples),
        "labels": list(combined_labels)
    }
    
    enhanced_file = str(PROCESSED_DIR / "enhanced_dataset_simple.json")
    with open(enhanced_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"增强数据集已保存到: {enhanced_file}")
    print(f"总样本数: {len(combined_samples)}")
    print(f"机器人数量: {sum(combined_labels)}")
    print(f"人类数量: {len(combined_labels) - sum(combined_labels)}")
    
    return enhanced_file

if __name__ == "__main__":
    # 创建增强数据集
    enhanced_file = create_enhanced_dataset_simple()
    
    print("\n数据集转换完成！")
    print(f"你可以使用以下命令训练模型:")
    print(f"python -m multiguard.training.train_model --data {enhanced_file}")
    print(f"或者在Web应用中上传 {enhanced_file} 文件进行训练") 