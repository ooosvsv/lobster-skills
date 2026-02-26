#!/usr/bin/env python3
"""Exa AI 搜索引擎 - 语义搜索增强版"""

import json
import urllib.request
import sys
import argparse
import os

EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
EXA_API_URL = "https://api.exa.ai/search"

def search_exa(query, num_results=10, category=None, livecrawl=False):
    """
    调用 Exa AI 搜索 API
    
    Args:
        query: 搜索查询
        num_results: 返回结果数量 (1-100)
        category: 搜索类别 (company, news, research paper, tweet, personal site, financial report)
        livecrawl: 是否实时抓取页面内容
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EXA_API_KEY}",
        "User-Agent": "curl/7.68.0"
    }
    
    data = {
        "query": query,
        "numResults": min(max(num_results, 1), 100),
        "type": "auto",
        "contents": {
            "text": True,
            "highlights": True
        }
    }
    
    if category:
        data["category"] = category
    
    if livecrawl:
        data["livecrawl"] = "always"
    
    req = urllib.request.Request(
        EXA_API_URL,
        data=json.dumps(data).encode('utf-8'),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode('utf-8')}"}
    except Exception as e:
        return {"error": str(e)}

def format_results(results):
    """格式化搜索结果为 Markdown"""
    if "error" in results:
        return f"❌ 搜索失败: {results['error']}"
    
    search_results = results.get("results", [])
    if not search_results:
        return "🤷 没有找到相关结果"
    
    output = [f"### 🔍 Exa 搜索结果 ({len(search_results)} 条)\n"]
    
    for i, result in enumerate(search_results, 1):
        title = result.get("title", "无标题")
        url = result.get("url", "")
        text = result.get("text", "")
        highlights = result.get("highlights", [])
        
        # 截断文本
        text_preview = text[:300] + "..." if len(text) > 300 else text
        
        output.append(f"**{i}. [{title}]({url})**")
        output.append(f"> {text_preview}\n")
        
        if highlights:
            output.append(f"💡 亮点: {highlights[0][:150]}...")
        
        output.append("")  # 空行
    
    return "\n".join(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exa AI 搜索")
    parser.add_argument("query", help="搜索查询")
    parser.add_argument("--num", type=int, default=10, help="结果数量")
    parser.add_argument("--category", choices=["company", "news", "research paper", "tweet", "personal site", "financial report"], help="搜索类别")
    parser.add_argument("--livecrawl", action="store_true", help="实时抓取页面")
    
    args = parser.parse_args()
    
    results = search_exa(args.query, args.num, args.category, args.livecrawl)
    print(format_results(results))
