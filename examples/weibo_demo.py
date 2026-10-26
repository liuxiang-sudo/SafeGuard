import requests
import webbrowser
import os
from multiguard.paths import REPORTS_DIR

BASE_URL = "http://127.0.0.1:5000/analyze"

# 模拟测试用户数据
test_users = [
    {
        "platform": "weibo",
        "username": "lily_chen",
        "email": "lily.chen88@example.com",
        "bio": "喜欢旅行和美食，摄影爱好者",
        "register_date": "2023-07-15",
        "ip_address": "61.218.145.23",
        "weibo_followers": "856",
        "weibo_following": "320",
        "weibo_posts": "198",
        "weibo_verified": "true",
        "weibo_last_posts": (
            "今天在京都拍到的樱花太美了\n"
            "刚学会做法式吐司\n"
            "周末去爬香山\n"
            "分享几张街拍\n"
            "这杯手冲咖啡真的绝"
        )
    },
    {
        "platform": "weibo",
        "username": "promo_sale_99999",
        "email": "promo99999@spam-mail.biz",
        "bio": "🔥自动化推广机器人🔥 全网最低价优惠，每日刷屏送福利！",
        "register_date": "2025-08-13",
        "ip_address": "203.0.113.250",
        "weibo_followers": "0",
        "weibo_following": "25000",
        "weibo_posts": "5000",
        "weibo_verified": "false",
        "weibo_last_posts": (
            "限时抢购！立即点击 http://spam.sale\n"
            "全场5折！猛戳链接领取优惠券 http://cheap.stuff\n"
            "买二送一，仅限今日 http://super.sale\n"
            "关注我，转发抽奖赢 iPhone\n"
            "🔥清仓特卖，最后一天！立即购买！🔥"
        )
    }
]

def main():
    for user in test_users:
        resp = requests.post(BASE_URL, data=user)
        if resp.ok:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            filename = REPORTS_DIR / f"result_{user['username']}.html"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(resp.text)
            print(f"✅ {user['username']} 提交成功，结果已保存到 {filename}")

            # 自动在默认浏览器打开
            file_path = os.path.abspath(filename)
            webbrowser.open(f"file:///{file_path}")
        else:
            print(f"❌ {user['username']} 提交失败，状态码: {resp.status_code}")


if __name__ == "__main__":
    main()
