from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
import pandas as pd
from multiguard.detection.platform_detector import analyze_platform_user
from multiguard.training.train_model import train_model_from_file
from multiguard.paths import DATASETS_DIR, WEB_DIR, ensure_runtime_dirs

app = Flask(__name__, template_folder=str(WEB_DIR / 'templates'),
            static_folder=str(WEB_DIR / 'static'))
app.config['SECRET_KEY'] = 'human-bot-detector-secret-key'
app.config['UPLOAD_FOLDER'] = str(DATASETS_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'json', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# 确保静态图像目录存在
ensure_runtime_dirs()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 获取平台类型
    platform = request.form.get('platform')
    
    # 基础数据（所有平台共有）
    user_data = {
        'platform': platform,
        'username': request.form.get('username', ''),
        'email': request.form.get('email', ''),
        'bio': request.form.get('bio', ''),
        'ip_address': request.form.get('ip_address', '') or request.remote_addr,
        'register_date': request.form.get('register_date', ''),
        'time_to_fill': request.form.get('time_to_fill', '0'),
        'typing_pattern': request.form.get('typing_pattern', ''),
        'navigation': request.form.get('navigation', ''),
        'screen_size': request.form.get('screen_size', ''),
        'user_agent': request.form.get('user_agent', ''),
        'referrer': request.form.get('referrer', ''),
        'click_pattern': request.form.get('click_pattern', '')
    }
    
    # 平台特定数据
    if platform == 'weibo':
        platform_data = {
            'followers': request.form.get('weibo_followers', ''),
            'following': request.form.get('weibo_following', ''),
            'posts': request.form.get('weibo_posts', ''),
            'verified': request.form.get('weibo_verified', 'none'),
            'last_posts': request.form.get('weibo_last_posts', '')
        }
        user_data.update(platform_data)
    elif platform == 'douyin':
        platform_data = {
            'followers': request.form.get('douyin_followers', ''),
            'following': request.form.get('douyin_following', ''),
            'likes': request.form.get('douyin_likes', ''),
            'videos': request.form.get('douyin_videos', ''),
            'verified': request.form.get('douyin_verified', 'false'),
            'video_titles': request.form.get('douyin_video_titles', '')
        }
        user_data.update(platform_data)
    elif platform == 'tieba':
        platform_data = {
            'level': request.form.get('tieba_level', ''),
            'age': request.form.get('tieba_age', ''),
            'posts': request.form.get('tieba_posts', ''),
            'followed': request.form.get('tieba_followed', ''),
            'last_posts': request.form.get('tieba_last_posts', '')
        }
        user_data.update(platform_data)
    
    # 使用平台特定分析
    result = analyze_platform_user(user_data)
    
    # 添加平台信息到结果
    result['platform'] = platform
    
    # 返回分析结果
    return render_template('result.html', result=result)

@app.route('/upload', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'dataset_file' not in request.files:
            flash('未找到文件', 'error')
            return redirect(request.url)
        
        file = request.files['dataset_file']
        
        # 如果用户未选择文件
        if file.filename == '':
            flash('未选择文件', 'error')
            return redirect(request.url)
        
        # 获取平台类型
        platform = request.form.get('platform')
        
        # 验证文件格式
        if file and allowed_file(file.filename):
            # 保存文件
            filename = secure_filename(file.filename)
            dataset_name = request.form.get('dataset_name', '')
            
            # 如果用户提供了数据集名称，使用它来命名文件
            if dataset_name:
                file_ext = filename.rsplit('.', 1)[1].lower()
                filename = f"{secure_filename(dataset_name)}_{platform}.{file_ext}"
            else:
                # 添加平台标识
                name_parts = filename.rsplit('.', 1)
                filename = f"{name_parts[0]}_{platform}.{name_parts[1]}"
                
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 检查是否需要与默认数据集混合
            mix_with_default = request.form.get('mix_with_default') == 'on'
            
            # 尝试训练模型
            try:
                # 调用训练函数，传递平台信息
                train_model_from_file(file_path, mix_with_default)
                flash(f'{platform} 平台数据集 {filename} 上传成功并已训练模型！', 'success')
            except Exception as e:
                flash(f'模型训练失败: {str(e)}', 'error')
                
            return redirect(url_for('index'))
        else:
            flash('不支持的文件格式，请上传 JSON 或 CSV 文件', 'error')
            return redirect(request.url)
            
    return render_template('upload.html')

@app.route('/download_template/<platform>')
def download_template(platform):
    # 根据不同平台提供不同的模板数据结构
    
    if platform == 'weibo':
        # 微博模板
        template_data = {
            "samples": [
                {
                    "platform": "weibo",
                    "username": "example_user123",
                    "email": "user@example.com",
                    "bio": "微博达人，热爱分享生活点滴",
                    "register_date": "2020-05-15",
                    "followers": "1250",
                    "following": "350",
                    "posts": "567",
                    "verified": "none",
                    "last_posts": "今天天气真好\n刚买了新手机，很喜欢\n准备去旅行了，好期待"
                }
            ],
            "labels": [0]  # 0 表示人类, 1 表示机器人
        }
    elif platform == 'douyin':
        # 抖音模板
        template_data = {
            "samples": [
                {
                    "platform": "douyin",
                    "username": "douyin_user888",
                    "email": "user@example.com",
                    "bio": "分享生活｜美食｜旅行",
                    "register_date": "2021-03-20",
                    "followers": "2800",
                    "following": "420",
                    "likes": "15600",
                    "videos": "45",
                    "verified": "false",
                    "video_titles": "如何在家做美味披萨\n巴厘岛旅行vlog\n这些小技巧让你的生活更便捷"
                }
            ],
            "labels": [0]  # 0 表示人类, 1 表示机器人
        }
    elif platform == 'tieba':
        # 贴吧模板
        template_data = {
            "samples": [
                {
                    "platform": "tieba",
                    "username": "tieba_fan123",
                    "email": "user@example.com",
                    "bio": "喜欢游戏和动漫，常活跃在相关贴吧",
                    "register_date": "2019-10-05",
                    "level": "8",
                    "age": "1236",
                    "posts": "342",
                    "followed": "15",
                    "last_posts": "请问这个游戏怎么过第三关？\n分享一下我的游戏装备\n求推荐好看的动漫"
                }
            ],
            "labels": [0]  # 0 表示人类, 1 表示机器人
        }
    elif platform == 'general':
        # 通用模板
        template_data = {
            "samples": [
                {
                    "platform": "general",
                    "username": "user123",
                    "email": "user@example.com",
                    "bio": "热爱生活，喜欢分享",
                    "register_date": "2022-01-01",
                    "ip_address": "127.0.0.1",
                    "time_to_fill": "12.5",
                    "typing_pattern": "fast",
                    "navigation": "首页->注册->填写信息",
                    "screen_size": "1920x1080",
                    "user_agent": "Mozilla/5.0",
                    "referrer": "https://example.com",
                    "click_pattern": "button1->input->submit"
                }
            ],
            "labels": [0]
        }
    else:
        # 未知平台，返回一个空模板或报错信息
        template_data = {
            "samples": [],
            "labels": []
        }
    
    return jsonify(template_data)

if __name__ == '__main__':
    app.run(debug=True) 