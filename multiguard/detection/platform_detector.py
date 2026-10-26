import re
import numpy as np
import datetime
import json
import os
from .ml_detector import ml_detector


def analyze_platform_user(user_data):
    """
    针对特定平台的用户数据进行分析，判断是机器人还是真人

    Args:
        user_data (dict): 包含用户数据的字典，包括平台特定信息

    Returns:
        dict: 包含分析结果的字典
    """
    platform = user_data.get('platform', 'general')

    # 尝试使用ML检测
    try:
        bot_probability, ml_evidence = ml_detector.predict_bot_probability(user_data)
        human_probability = 1 - bot_probability

        # 增加平台特定分析
        platform_bot_prob, platform_evidence = analyze_platform_specific(user_data)

        # 综合ML和平台特定分析，权重比例70:30
        final_bot_prob = bot_probability * 0.7 + platform_bot_prob * 0.3
        final_human_prob = 1 - final_bot_prob

        # 合并证据
        combined_evidence = ml_evidence + platform_evidence

        # 确定分类
        if final_human_prob >= 0.6:
            classification = "Human"
        elif final_human_prob <= 0.4:
            classification = "Bot"
        else:
            classification = "Uncertain"

        return {
            'probability_human': round(final_human_prob, 2),
            'probability_bot': round(final_bot_prob, 2),
            'classification': classification,
            'evidence': combined_evidence
        }
    except Exception as e:
        # 如果ML检测失败，回退到平台特定分析
        print(f"ML detection failed: {str(e)}. Using platform-specific analysis only.")
        return platform_specific_fallback(user_data)


def platform_specific_fallback(user_data):
    """
    当ML检测失败时使用的平台特定分析回退方法

    Args:
        user_data (dict): 用户数据字典

    Returns:
        dict: 包含分析结果的字典
    """
    platform = user_data.get('platform', 'general')

    platform_bot_prob, platform_evidence = analyze_platform_specific(user_data)
    human_probability = 1 - platform_bot_prob

    # 确定分类
    if human_probability >= 0.7:
        classification = "Human"
    elif human_probability <= 0.3:
        classification = "Bot"
    else:
        classification = "Uncertain"

    platform_evidence.append(f"注: 分析仅使用{get_platform_name(platform)}平台特定规则进行")

    return {
        'probability_human': round(human_probability, 2),
        'probability_bot': round(platform_bot_prob, 2),
        'classification': classification,
        'evidence': platform_evidence
    }


def analyze_platform_specific(user_data):
    """
    根据不同平台选择不同的分析方法

    Args:
        user_data (dict): 用户数据字典

    Returns:
        tuple: (机器人概率, 证据列表)
    """
    platform = user_data.get('platform', 'general')

    if platform == 'weibo':
        return analyze_weibo_user(user_data)
    elif platform == 'douyin':
        return analyze_douyin_user(user_data)
    elif platform == 'tieba':
        return analyze_tieba_user(user_data)
    else:
        # 如果是未知平台或general，使用基本分析
        return analyze_general_user(user_data)


def get_platform_name(platform):
    """获取平台的中文名称"""
    platform_names = {
        'general': '初步检测',
        'weibo': '微博',
        'douyin': '抖音',
        'tieba': '贴吧'
    }
    return platform_names.get(platform, '未知平台')


def analyze_account_age(register_date):
    """
    分析账号年龄，新账号更可能是机器人

    Args:
        register_date (str): 注册日期字符串，格式为YYYY-MM-DD

    Returns:
        tuple: (评分, 证据)
    """
    evidence = []

    if not register_date:
        evidence.append("未提供注册日期信息")
        return 0.5, evidence

    try:
        reg_date = datetime.datetime.strptime(register_date, "%Y-%m-%d").date()
        today = datetime.datetime.now().date()
        account_age = (today - reg_date).days

        if account_age < 7:
            evidence.append(f"账号注册时间仅{account_age}天，非常新的账号通常与机器人关联")
            return 0.2, evidence
        elif account_age < 30:
            evidence.append(f"账号注册时间为{account_age}天，较新的账号有时与自动化行为相关")
            return 0.4, evidence
        elif account_age < 180:
            evidence.append(f"账号注册时间为{account_age}天，中等账龄")
            return 0.6, evidence
        else:
            evidence.append(f"账号注册时间为{account_age}天，长期活跃的账号更可能是人类用户")
            return 0.8, evidence
    except:
        evidence.append("注册日期格式无效，无法分析账号年龄")
        return 0.5, evidence


def analyze_follower_ratio(followers, following):
    """
    分析粉丝-关注比例

    Args:
        followers (str): 粉丝数量
        following (str): 关注数量

    Returns:
        tuple: (评分, 证据)
    """
    evidence = []

    try:
        followers_count = int(followers) if followers else 0
        following_count = int(following) if following else 0

        if followers_count == 0 and following_count == 0:
            evidence.append("账号没有粉丝和关注，通常表示新账号或不活跃账号")
            return 0.4, evidence

        if following_count == 0:
            evidence.append("账号有粉丝但没有关注任何人，这是不寻常的行为模式")
            return 0.3, evidence

        ratio = followers_count / following_count if following_count > 0 else 0

        if ratio > 50:
            evidence.append(f"粉丝与关注比例异常高 ({ratio:.1f})，可能是名人账号或异常账号")
            return 0.6, evidence
        elif ratio > 10:
            evidence.append(f"粉丝与关注比例较高 ({ratio:.1f})，可能是较有影响力的账号")
            return 0.7, evidence
        elif ratio > 0.1:
            evidence.append(f"粉丝与关注比例正常 ({ratio:.1f})，符合典型人类用户")
            return 0.8, evidence
        else:
            evidence.append(f"关注数远高于粉丝数 ({ratio:.1f})，常见于机器人批量关注行为")
            return 0.2, evidence
    except:
        evidence.append("无法分析粉丝-关注比例，数据格式可能无效")
        return 0.5, evidence


def analyze_content_pattern(content, content_type="文本"):
    """
    分析内容模式，检测是否有重复、批量生成的特征

    Args:
        content (str): 内容文本，多条内容用换行分隔
        content_type (str): 内容类型描述

    Returns:
        tuple: (评分, 证据)
    """
    evidence = []

    if not content:
        evidence.append(f"未提供{content_type}内容")
        return 0.5, evidence

    # 分割多条内容
    items = content.strip().split('\n')
    items = [item.strip() for item in items if item.strip()]

    if not items:
        evidence.append(f"无有效{content_type}内容")
        return 0.5, evidence

    # 检查内容长度
    avg_length = sum(len(item) for item in items) / len(items)

    # 检查重复模式
    unique_items = set(items)
    if len(unique_items) < len(items) * 0.8:
        evidence.append(f"{content_type}内容中存在明显重复，可能是自动生成")
        duplication_score = 0.3
    else:
        duplication_score = 0.7

    # 检查模板化内容，如数字序列或类似格式
    pattern_counts = 0
    for item in items:
        if re.search(r'\d{5,}', item) or re.search(r'(第\d+|[a-zA-Z]{5,})', item):
            pattern_counts += 1

    pattern_ratio = pattern_counts / len(items) if items else 0
    if pattern_ratio > 0.5:
        evidence.append(f"{content_type}内容显示出模板化模式，常见于自动生成内容")
        pattern_score = 0.3
    else:
        pattern_score = 0.8

    # 评估内容多样性
    if avg_length < 5:
        evidence.append(f"{content_type}内容平均长度过短 ({avg_length:.1f}字符)，不太自然")
        length_score = 0.4
    elif avg_length > 100:
        evidence.append(f"{content_type}内容平均长度较长 ({avg_length:.1f}字符)，通常表示更多人工编写")
        length_score = 0.8
    else:
        evidence.append(f"{content_type}内容平均长度适中 ({avg_length:.1f}字符)")
        length_score = 0.6

    # 综合评分
    final_score = (duplication_score + pattern_score + length_score) / 3

    if len(items) > 5:
        evidence.append(f"提供了{len(items)}条{content_type}内容，数量丰富有助于分析")

    return final_score, evidence


def analyze_general_user(user_data):
    """
    分析通用平台的用户数据

    Args:
        user_data (dict): 用户数据字典

    Returns:
        tuple: (机器人概率, 证据列表)
    """
    evidence = []
    scores = []

    # 分析账号年龄
    age_score, age_evidence = analyze_account_age(user_data.get('register_date', ''))
    scores.append(age_score)
    evidence.extend(age_evidence)

    # 分析用户代理
    ua = user_data.get('user_agent', '')
    if 'bot' in ua.lower() or 'crawler' in ua.lower() or 'spider' in ua.lower():
        evidence.append(f"用户代理字符串包含机器人相关标识: {ua}")
        scores.append(0.2)
    elif 'mozilla' in ua.lower() and ('chrome' in ua.lower() or 'safari' in ua.lower() or 'firefox' in ua.lower()):
        evidence.append("用户代理字符串表示标准浏览器，常见于人类用户")
        scores.append(0.7)

    # 分析点击模式
    click_pattern = user_data.get('click_pattern', '')
    if click_pattern:
        clicks = click_pattern.count('|') + 1
        if clicks > 5:
            evidence.append(f"检测到{clicks}次点击事件，表示正常的人类交互")
            scores.append(0.8)
        else:
            evidence.append(f"仅检测到{clicks}次点击事件，互动较少")
            scores.append(0.5)

    # 设备信息
    screen_size = user_data.get('screen_size', '')
    if screen_size:
        evidence.append(f"使用设备屏幕分辨率: {screen_size}")
        # 移动设备更可能是人类用户
        if 'x' in screen_size:
            width = int(screen_size.split('x')[0])
            if width < 900:
                evidence.append("检测到移动设备屏幕尺寸，更常见于人类用户")
                scores.append(0.7)

    # 平均评分
    avg_score = sum(scores) / len(scores) if scores else 0.5

    # 转换为机器人概率
    bot_probability = 1 - avg_score

    return bot_probability, evidence


def analyze_weibo_user(user_data):
    """
    分析微博用户数据，判断是否为机器人

    Args:
        user_data (dict): 用户数据字典

    Returns:
        tuple: (机器人概率, 证据列表)
    """
    evidence = []
    scores = []

    # 获取微博特定数据
    followers = user_data.get('followers', '')
    following = user_data.get('following', '')
    posts = user_data.get('posts', '')
    verified = user_data.get('verified', 'none')
    last_posts = user_data.get('last_posts', '')

    # 分析注册时间
    age_score, age_evidence = analyze_account_age(user_data.get('register_date', ''))
    scores.append(age_score)
    evidence.extend(age_evidence)

    # 分析粉丝-关注比例
    follower_score, follower_evidence = analyze_follower_ratio(followers, following)
    scores.append(follower_score)
    evidence.extend(follower_evidence)

    # 分析发帖数量与账号年龄的关系
    try:
        posts_count = int(posts) if posts else 0
        reg_date = datetime.datetime.strptime(user_data.get('register_date', ''), "%Y-%m-%d").date()
        today = datetime.datetime.now().date()
        account_days = (today - reg_date).days

        if account_days > 0:
            posts_per_day = posts_count / account_days

            if posts_per_day > 20:
                evidence.append(f"平均每天发布{posts_per_day:.1f}条微博，发布频率异常高，常见于自动发布账号")
                scores.append(0.2)
            elif posts_per_day > 10:
                evidence.append(f"平均每天发布{posts_per_day:.1f}条微博，发布频率较高")
                scores.append(0.4)
            elif posts_per_day > 0.5:
                evidence.append(f"平均每天发布{posts_per_day:.1f}条微博，发布频率适中，符合活跃人类用户")
                scores.append(0.8)
            elif posts_per_day > 0.1:
                evidence.append(f"平均每天发布{posts_per_day:.1f}条微博，发布频率较低")
                scores.append(0.7)
            else:
                evidence.append(f"平均每天发布{posts_per_day:.1f}条微博，几乎不发布内容")
                scores.append(0.5)
    except:
        # 如果无法计算，不影响其他分析
        pass

    # 分析认证状态
    if verified == 'personal':
        evidence.append("账号已通过个人认证，显著降低了其为机器人的可能性")
        scores.append(0.9)
    elif verified == 'business':
        evidence.append("账号已通过企业认证，但企业账号有时会使用自动化工具")
        scores.append(0.7)

    # 分析微博内容
    content_score, content_evidence = analyze_content_pattern(last_posts, "微博")
    scores.append(content_score)
    evidence.extend(content_evidence)

    # 检查微博内容中的广告特征
    if last_posts:
        posts_list = last_posts.strip().split('\n')
        ad_keywords = ['免费领取', '点击链接', '推广', '限时优惠', '抢购', '下单立减',
                       '官方旗舰店', '优惠券', '转发抽奖', '抽iPhone']

        ad_count = 0
        for post in posts_list:
            if any(keyword in post for keyword in ad_keywords):
                ad_count += 1

        ad_ratio = ad_count / len(posts_list) if posts_list else 0
        if ad_ratio > 0.5:
            evidence.append(f"微博内容中广告特征明显，{ad_count}条微博中包含广告关键词，常见于营销机器人")
            scores.append(0.2)
        elif ad_ratio > 0.3:
            evidence.append(f"微博内容包含一些广告特征，{ad_count}条微博中包含广告关键词")
            scores.append(0.4)

    # 微博特定的异常模式检测
    username = user_data.get('username', '')
    if username and re.match(r'^[a-zA-Z0-9_-]+\d{5,}$', username):
        evidence.append(f"用户名'{username}'末尾包含5位以上数字，这是微博机器人常见的命名模式")
        scores.append(0.3)

    # 平均评分
    avg_score = sum(scores) / len(scores) if scores else 0.5

    # 转换为机器人概率
    bot_probability = 1 - avg_score

    # 添加平台特定说明
    evidence.insert(0, f"基于微博平台特定的行为和内容模式分析")

    return bot_probability, evidence


def analyze_douyin_user(user_data):
    """
    分析抖音用户数据，判断是否为机器人

    Args:
        user_data (dict): 用户数据字典

    Returns:
        tuple: (机器人概率, 证据列表)
    """
    evidence = []
    scores = []

    # 获取抖音特定数据
    followers = user_data.get('followers', '')
    following = user_data.get('following', '')
    likes = user_data.get('likes', '')
    videos = user_data.get('videos', '')
    verified = user_data.get('verified', 'false')
    video_titles = user_data.get('video_titles', '')

    # 分析注册时间
    age_score, age_evidence = analyze_account_age(user_data.get('register_date', ''))
    scores.append(age_score)
    evidence.extend(age_evidence)

    # 分析粉丝-关注比例
    follower_score, follower_evidence = analyze_follower_ratio(followers, following)
    scores.append(follower_score)
    evidence.extend(follower_evidence)

    # 分析抖音特有的指标：获赞数与视频数的比例
    try:
        likes_count = int(likes) if likes else 0
        videos_count = int(videos) if videos else 0

        if videos_count > 0:
            likes_per_video = likes_count / videos_count

            if likes_per_video > 10000:
                evidence.append(f"平均每个视频获赞{likes_per_video:.0f}次，获赞率极高，可能是热门创作者或使用了刷赞服务")
                scores.append(0.5)  # 高获赞率对人/机器人判断是中性的
            elif likes_per_video > 1000:
                evidence.append(f"平均每个视频获赞{likes_per_video:.0f}次，获赞率较高，内容质量好")
                scores.append(0.7)
            elif likes_per_video > 100:
                evidence.append(f"平均每个视频获赞{likes_per_video:.0f}次，获赞率正常")
                scores.append(0.8)
            elif likes_per_video > 10:
                evidence.append(f"平均每个视频获赞{likes_per_video:.0f}次，获赞率较低")
                scores.append(0.6)
            else:
                evidence.append(f"平均每个视频获赞{likes_per_video:.0f}次，获赞率很低，可能是低质量内容或机器人")
                scores.append(0.3)
    except:
        # 如果无法计算，不影响其他分析
        pass

    # 分析视频发布频率
    try:
        videos_count = int(videos) if videos else 0
        reg_date = datetime.datetime.strptime(user_data.get('register_date', ''), "%Y-%m-%d").date()
        today = datetime.datetime.now().date()
        account_days = (today - reg_date).days

        if account_days > 0:
            videos_per_day = videos_count / account_days

            if videos_per_day > 5:
                evidence.append(f"平均每天发布{videos_per_day:.1f}个视频，发布频率异常高，可能使用自动化工具")
                scores.append(0.2)
            elif videos_per_day > 2:
                evidence.append(f"平均每天发布{videos_per_day:.1f}个视频，发布频率较高，可能是专职创作者")
                scores.append(0.6)
            elif videos_per_day > 0.5:
                evidence.append(f"平均每天发布{videos_per_day:.1f}个视频，发布频率适中，符合活跃的人类用户")
                scores.append(0.8)
            elif videos_per_day > 0.1:
                evidence.append(f"平均每天发布{videos_per_day:.1f}个视频，发布频率较低")
                scores.append(0.7)
            else:
                evidence.append(f"平均每天发布{videos_per_day:.1f}个视频，几乎不发布视频")
                scores.append(0.5)
    except:
        # 如果无法计算，不影响其他分析
        pass

    # 分析认证状态
    if verified == 'true':
        evidence.append("账号已认证，这降低了其为机器人的可能性")
        scores.append(0.8)

    # 分析视频标题内容
    content_score, content_evidence = analyze_content_pattern(video_titles, "视频标题")
    scores.append(content_score)
    evidence.extend(content_evidence)

    # 抖音特定的异常模式：检测热门标签堆砌
    if video_titles:
        titles_list = video_titles.strip().split('\n')
        hashtag_count = 0
        for title in titles_list:
            hashtags = re.findall(r'#\w+', title)
            if len(hashtags) > 5:
                hashtag_count += 1

        hashtag_ratio = hashtag_count / len(titles_list) if titles_list else 0
        if hashtag_ratio > 0.5:
            evidence.append(f"{hashtag_count}个视频标题中有大量标签堆砌现象，这是抖音机器人常见的优化方式")
            scores.append(0.3)

    # 抖音特定的机器人标识：判断是否是营销号
    if video_titles:
        titles_list = video_titles.strip().split('\n')
        marketing_keywords = ['带货', '优惠券', '商品链接', '购买链接', '团购', '秒杀',
                              '专卖', '特卖', '折扣', '降价', '买一送一', '黑五']

        marketing_count = 0
        for title in titles_list:
            if any(keyword in title for keyword in marketing_keywords):
                marketing_count += 1

        marketing_ratio = marketing_count / len(titles_list) if titles_list else 0
        if marketing_ratio > 0.7:
            evidence.append(f"视频标题中营销内容比例高达{marketing_ratio * 100:.0f}%，典型的营销账号特征")
            scores.append(0.4)
        elif marketing_ratio > 0.4:
            evidence.append(f"视频标题中包含较多营销内容，占比{marketing_ratio * 100:.0f}%")
            scores.append(0.5)

    # 用户名检测
    username = user_data.get('username', '')
    if username and (re.match(r'^[a-zA-Z0-9_-]+\d{6,}$', username) or
                     re.match(r'^抖音用户[a-zA-Z0-9]+$', username)):
        evidence.append(f"用户名'{username}'符合抖音平台自动生成账号的典型模式")
        scores.append(0.3)

    # 平均评分
    avg_score = sum(scores) / len(scores) if scores else 0.5

    # 转换为机器人概率
    bot_probability = 1 - avg_score

    # 添加平台特定说明
    evidence.insert(0, f"基于抖音平台特定的行为和内容模式分析")

    return bot_probability, evidence


def analyze_tieba_user(user_data):
    """
    分析贴吧用户数据，判断是否为机器人

    Args:
        user_data (dict): 用户数据字典

    Returns:
        tuple: (机器人概率, 证据列表)
    """
    evidence = []
    scores = []

    # 获取贴吧特定数据
    level = user_data.get('level', '')
    age = user_data.get('age', '')
    posts = user_data.get('posts', '')
    followed = user_data.get('followed', '')
    last_posts = user_data.get('last_posts', '')

    # 分析注册时间（吧龄）
    try:
        tieba_age = int(age) if age else 0

        if tieba_age < 7:
            evidence.append(f"吧龄仅{tieba_age}天，非常新的贴吧账号通常与批量注册机器人相关")
            scores.append(0.2)
        elif tieba_age < 30:
            evidence.append(f"吧龄为{tieba_age}天，较新的贴吧账号")
            scores.append(0.4)
        elif tieba_age < 365:
            evidence.append(f"吧龄为{tieba_age}天，中等吧龄")
            scores.append(0.6)
        else:
            evidence.append(f"吧龄为{tieba_age}天，老账号，吧龄较长的贴吧账号更可能是人类用户")
            scores.append(0.8)
    except:
        evidence.append("未提供有效的吧龄信息")
        scores.append(0.5)

    # 分析等级与吧龄的关系
    try:
        tieba_level = int(level) if level else 0
        tieba_age = int(age) if age else 0

        if tieba_age > 0:
            level_per_year = tieba_level / (tieba_age / 365)

            if level_per_year > 10:
                evidence.append(f"账号等级增长异常快 (每年{level_per_year:.1f}级)，可能使用了刷级工具")
                scores.append(0.3)
            elif level_per_year > 5:
                evidence.append(f"账号等级增长较快 (每年{level_per_year:.1f}级)，表示高度活跃")
                scores.append(0.7)
            elif level_per_year > 2:
                evidence.append(f"账号等级增长正常 (每年{level_per_year:.1f}级)，符合典型贴吧用户")
                scores.append(0.8)
            else:
                evidence.append(f"账号等级增长缓慢 (每年{level_per_year:.1f}级)，活跃度较低")
                scores.append(0.6)
    except:
        # 如果无法计算，不影响其他分析
        pass

    # 分析发帖量与吧龄的关系
    try:
        posts_count = int(posts) if posts else 0
        tieba_age = int(age) if age else 0

        if tieba_age > 0:
            posts_per_day = posts_count / tieba_age

            if posts_per_day > 10:
                evidence.append(f"平均每天发布{posts_per_day:.1f}个帖子，发帖频率异常高，可能是自动发帖机器人")
                scores.append(0.2)
            elif posts_per_day > 5:
                evidence.append(f"平均每天发布{posts_per_day:.1f}个帖子，发帖频率很高")
                scores.append(0.4)
            elif posts_per_day > 1:
                evidence.append(f"平均每天发布{posts_per_day:.1f}个帖子，发帖频率较高，是活跃用户")
                scores.append(0.7)
            elif posts_per_day > 0.2:
                evidence.append(f"平均每天发布{posts_per_day:.1f}个帖子，发帖频率正常")
                scores.append(0.8)
            else:
                evidence.append(f"平均每天发布{posts_per_day:.1f}个帖子，发帖频率较低")
                scores.append(0.6)
    except:
        # 如果无法计算，不影响其他分析
        pass

    # 分析关注的贴吧数量
    try:
        followed_count = int(followed) if followed else 0

        if followed_count == 0:
            evidence.append("未关注任何贴吧，是不寻常的行为")
            scores.append(0.4)
        elif followed_count < 3:
            evidence.append(f"仅关注了{followed_count}个贴吧，比较少见")
            scores.append(0.5)
        elif followed_count < 20:
            evidence.append(f"关注了{followed_count}个贴吧，符合典型贴吧用户")
            scores.append(0.8)
        elif followed_count < 100:
            evidence.append(f"关注了{followed_count}个贴吧，表示是贴吧重度用户")
            scores.append(0.7)
        else:
            evidence.append(f"关注了{followed_count}个贴吧，数量异常多，可能是批量关注行为")
            scores.append(0.3)
    except:
        # 如果无法计算，不影响其他分析
        pass

    # 分析帖子内容
    content_score, content_evidence = analyze_content_pattern(last_posts, "帖子")
    scores.append(content_score)
    evidence.extend(content_evidence)

    # 贴吧特定的机器人识别：检测广告和灌水
    if last_posts:
        posts_list = last_posts.strip().split('\n')
        ad_keywords = ['微信', '加群', '优惠券', 'Q群', '代购', '点击链接', '私聊',
                       '联系方式', '扣扣', '赚钱', '兼职', '招聘']

        ad_count = 0
        for post in posts_list:
            if any(keyword in post for keyword in ad_keywords):
                ad_count += 1

        ad_ratio = ad_count / len(posts_list) if posts_list else 0
        if ad_ratio > 0.5:
            evidence.append(f"帖子内容中广告特征明显，{ad_count}条帖子包含广告关键词，贴吧典型的广告机器人特征")
            scores.append(0.2)
        elif ad_ratio > 0.3:
            evidence.append(f"帖子内容包含一些广告特征，{ad_count}条帖子包含广告关键词")
            scores.append(0.4)

    # 分析帖子长度分布
    if last_posts:
        posts_list = last_posts.strip().split('\n')
        short_post_count = len([p for p in posts_list if len(p) < 10])

        if short_post_count > len(posts_list) * 0.7:
            evidence.append(f"{short_post_count}条帖子过短 (少于10个字符)，符合贴吧灌水机器人特征")
            scores.append(0.3)

    # 用户名分析
    username = user_data.get('username', '')
    if username and (re.match(r'^[a-zA-Z0-9_]+\d{7,}$', username) or
                     re.match(r'^百度用户_[a-zA-Z0-9]+$', username)):
        evidence.append(f"用户名'{username}'符合贴吧平台自动生成账号的典型模式")
        scores.append(0.3)

    # 平均评分
    avg_score = sum(scores) / len(scores) if scores else 0.5

    # 转换为机器人概率
    bot_probability = 1 - avg_score

    # 添加平台特定说明
    evidence.insert(0, f"基于贴吧平台特定的行为和内容模式分析")

    return bot_probability, evidence
