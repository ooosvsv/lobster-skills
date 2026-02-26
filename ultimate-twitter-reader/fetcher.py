import json
import urllib.request
import re
import argparse
import hashlib
from datetime import datetime, timezone

def fetch_with_fallback(tweet_id):
    """多端点容灾：首选 vxtwitter，失败无缝切换 fxtwitter"""
    endpoints = [
        f"https://api.vxtwitter.com/Twitter/status/{tweet_id}",
        f"https://api.fxtwitter.com/Twitter/status/{tweet_id}"
    ]
    for api_url in endpoints:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception:
            continue
    return {"error": "推文可能已被删除、设置为私密，或端点暂时不可用。"}

def format_deepreeder_md(data, original_url):
    """融合 DeepReeder 风格，生成结构化 Markdown"""
    if "error" in data:
        return f"❌ [抓取失败]({original_url}): {data['error']}\n"
    
    text = data.get("text", "")
    author_name = data.get("user_name", "Unknown")
    author_handle = data.get("user_screen_name", "unknown")
    date_str = data.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    likes = data.get("likes", 0)
    retweets = data.get("retweets", 0)
    content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]
    
    # 构建 YAML 元数据和正文
    md = f"""---
author: "@{author_handle}"
source: "{original_url}"
date: "{date_str}"
content_hash: "{content_hash}"
---

### 🐦 {author_name} (@{author_handle})
🕒 时间: {date_str} | 📊 互动: ❤️ {likes} · 🔁 {retweets}

> {text.replace(chr(10), chr(10) + '> ')}
"""
    
    # 处理媒体：格式化为标准 Markdown 图片语法，以便大模型触发 Vision 能力
    media_list = data.get("mediaURLs", [])
    if media_list:
        md += "\n**📸 附带媒体 (Media):**\n"
        for i, url in enumerate(media_list):
            if "video" in url or ".mp4" in url:
                md += f"- 🎥 [点击查看视频源文件]({url})\n"
            else:
                md += f"![Image_{i}]({url})\n"
    
    return md

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate Twitter Fetcher")
    parser.add_argument("--urls", required=True, help="包含 X/Twitter 链接的一段文本")
    args = parser.parse_args()
    
    # 使用正则一次性提取输入文本中的所有推文 ID（支持批量抓取）
    tweet_ids = set(re.findall(r'(?:twitter\\.com|x\\.com)/[a-zA-Z0-9_]+/status/([0-9]+)', args.urls))
    
    if not tweet_ids:
        print("⚠️ 未在输入中检测到有效的 Twitter/X 链接。")
        exit(1)
    
    print("### 🔍 抓取结果\n")
    for tid in tweet_ids:
        url = f"https://x.com/i/status/{tid}"
        data = fetch_with_fallback(tid)
        print(format_deepreeder_md(data, url))
        print("\n---\n")
