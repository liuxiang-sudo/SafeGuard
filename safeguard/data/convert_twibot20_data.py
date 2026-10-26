import json
from safeguard.paths import TWIBOT_DIR, PROCESSED_DIR, ensure_runtime_dirs
import random
import os
from datetime import datetime, timedelta
import re

def convert_twibot20_to_system_format():
    """
    将TwiBot-20数据转换为适合现有系统的格式
    """
    print("开始转换TwiBot-20数据...")
    
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
            user_id = user['ID']
            profile = user['profile'] or {}
            tweets = user['tweet'] or []
            domain = user['domain']
            
            # 生成用户名
            if profile.get('screen_name'):
                username = profile['screen_name']
            else:
                username = f"user_{user_id}"
            
            # 生成邮箱
            email = f"{username.lower().replace(' ', '_')}@example.com"
            
            # 提取bio（用户描述）
            bio = ""
            if profile.get('description'):
                bio = profile['description']
            elif tweets and len(tweets) > 0:
                # 如果没有描述，使用第一条推文作为bio
                first_tweet = tweets[0]
                bio = first_tweet[:200] if len(first_tweet) > 200 else first_tweet
            
            # 如果没有bio，生成一个基于领域的
            if not bio:
                domain_bios = {
                    'politics': '关注政治动态，分享时事观点',
                    'business': '商业分析师，关注市场趋势',
                    'entertainment': '娱乐爱好者，分享有趣内容',
                    'sports': '体育迷，热爱各种运动'
                }
                bio = domain_bios.get(domain, '热爱生活，喜欢分享')
            
            # 生成注册日期
            if profile.get('created_at'):
                try:
                    # 解析Twitter日期格式
                    created_date = datetime.strptime(
                        profile['created_at'], 
                        '%a %b %d %X %z %Y'
                    )
                    register_date = created_date.strftime('%Y-%m-%d')
                except:
                    register_date = "2020-01-01"
            else:
                register_date = "2020-01-01"
            
            # 生成IP地址
            ip_address = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
            
            # 根据用户特征判断是否为机器人
            # 使用一些启发式规则来判断
            is_bot = determine_if_bot(profile, tweets, domain)
            
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
    
    output_file = str(PROCESSED_DIR / "twibot20_converted.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换后的数据已保存到: {output_file}")
    
    return output_file

def determine_if_bot(profile, tweets, domain):
    """
    使用启发式规则判断用户是否为机器人
    """
    bot_score = 0
    
    # 检查关注者比例 - 确保转换为数值类型
    try:
        followers = int(profile.get('followers_count', 0))
        friends = int(profile.get('friends_count', 0))
        
        if followers > 0 and friends > 0:
            ratio = followers / friends
            if ratio > 10:  # 关注者远多于关注的人
                bot_score += 2
            elif ratio < 0.1:  # 关注的人远多于关注者
                bot_score += 1
    except (ValueError, TypeError):
        pass
    
    # 检查推文数量
    try:
        statuses = int(profile.get('statuses_count', 0))
        if statuses > 10000:  # 推文过多
            bot_score += 1
    except (ValueError, TypeError):
        pass
    
    # 检查是否验证
    if not profile.get('verified', False):
        bot_score += 0.5
    
    # 检查用户名特征
    screen_name = profile.get('screen_name', '')
    if screen_name:
        # 检查是否包含机器人相关词汇
        bot_keywords = ['bot', 'auto', 'ai', 'robot', 'official', 'service', 'system']
        if any(keyword in screen_name.lower() for keyword in bot_keywords):
            bot_score += 2
        
        # 检查是否包含大量数字
        digit_count = sum(c.isdigit() for c in screen_name)
        if digit_count > len(screen_name) * 0.5:
            bot_score += 1
    
    # 检查描述特征
    description = profile.get('description', '')
    if description:
        # 检查是否包含链接
        if 'http' in description.lower():
            bot_score += 1
        
        # 检查是否包含机器人相关词汇
        bot_keywords = ['bot', 'auto', 'ai', 'robot', 'official', 'service', 'system']
        if any(keyword in description.lower() for keyword in bot_keywords):
            bot_score += 2
        
        # 检查描述长度
        if len(description) < 10 or len(description) > 500:
            bot_score += 0.5
    
    # 检查推文特征
    if tweets and len(tweets) > 0:
        # 检查推文内容是否重复 - 将推文转换为字符串后再去重
        tweet_strings = [str(tweet) for tweet in tweets]
        unique_tweets = set(tweet_strings)
        if len(unique_tweets) < len(tweet_strings) * 0.8:
            bot_score += 2
        
        # 检查推文是否包含大量链接
        link_count = sum(1 for tweet in tweet_strings if 'http' in tweet.lower())
        if link_count > len(tweet_strings) * 0.3:
            bot_score += 1
    
    # 根据领域调整分数
    domain_weights = {
        'politics': 0.5,  # 政治领域可能有更多机器人
        'business': 0.3,  # 商业领域
        'entertainment': 0.2,  # 娱乐领域
        'sports': 0.1  # 体育领域
    }
    domain_weight = domain_weights.get(domain, 0.2)
    bot_score *= domain_weight
    
    # 判断是否为机器人（分数阈值）
    return bot_score >= 1.5

def create_enhanced_dataset():
    """
    创建增强数据集，结合TwiBot-20数据和现有合成数据
    """
    print("创建增强数据集...")
    
    # 转换TwiBot-20数据
    twibot_file = convert_twibot20_to_system_format()
    
    # 加载转换后的数据
    with open(twibot_file, 'r', encoding='utf-8') as f:
        twibot_data = json.load(f)
    
    # 生成额外的合成数据以平衡数据集
    from safeguard.training.train_model import generate_synthetic_data
    
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
    
    enhanced_file = str(PROCESSED_DIR / "enhanced_dataset.json")
    with open(enhanced_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"增强数据集已保存到: {enhanced_file}")
    print(f"总样本数: {len(combined_samples)}")
    print(f"机器人数量: {sum(combined_labels)}")
    print(f"人类数量: {len(combined_labels) - sum(combined_labels)}")
    
    return enhanced_file

if __name__ == "__main__":
    # 创建增强数据集
    enhanced_file = create_enhanced_dataset()
    
    print("\n数据集转换完成！")
    print(f"你可以使用以下命令训练模型:")
    print(f"python -m safeguard.training.train_model --data {enhanced_file}")
    print(f"或者在Web应用中上传 {enhanced_file} 文件进行训练") 