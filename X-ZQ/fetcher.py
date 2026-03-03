import argparse
import hashlib
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone


def extract_urls(text: str):
    """从输入文本中提取 X/Twitter 链接"""
    urls = re.findall(r"https?://(?:x|twitter)\.com/[^\s]+", text)
    cleaned = []
    for u in urls:
        u = u.rstrip('),.;!\"\'')
        if u not in cleaned:
            cleaned.append(u)
    return cleaned


def run_cmd(cmd, input_text=None, timeout=120):
    p = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip())
    return p.stdout.strip()


def parse_eval_output(raw: str):
    """agent-browser eval 输出容错解析"""
    s = raw.strip()
    if not s:
        return {}

    # 先尝试直接 JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 再尝试去掉外层字符串转义
    try:
        unescaped = json.loads(s)
        if isinstance(unescaped, str):
            return json.loads(unescaped)
    except Exception:
        pass

    return {"text": s}


def clean_text(raw: str):
    if not raw:
        return ""

    lines = [ln.strip() for ln in raw.splitlines()]

    # 截断明显无关尾部
    cut_markers = [
        "Want to publish your own Article?",
        "New to X?",
        "Trending now",
        "Terms of Service",
    ]

    noise_lines = {
        "Don’t miss what’s happening",
        "People on X are the first to know.",
        "Log in",
        "Sign up",
        "Article",
        "See new posts",
        "Conversation",
        "What’s happening",
        "Show more",
        "Privacy Policy",
        "Cookie Policy",
        "Accessibility",
        "Ads info",
        "More",
    }

    cleaned = []
    for ln in lines:
        if not ln:
            continue

        # 顶部/底部常见噪音
        if ln in noise_lines:
            continue

        # 过滤明显互动计数噪音（纯数字/带K计数）
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[KMB]?", ln):
            continue

        if any(ln.startswith(m) for m in cut_markers):
            break

        cleaned.append(ln)

    out = "\n".join(cleaned).strip()
    # 清洗过度时回退原文，避免误判“空正文”
    if len(out) < 200 and len(raw.strip()) > len(out):
        return raw.strip()
    return out


def clean_media_urls(urls):
    """过滤头像/emoji/站点图标等无关媒体，保留正文相关图视频"""
    if not urls:
        return []

    ignored_patterns = [
        r"emoji\.twimg\.com",
        r"abs-\d+\.twimg\.com/emoji/",
        r"profile_images/",
        r"/favicon",
        r"\.svg(?:\?|$)",
    ]

    out = []
    for u in urls:
        if not u:
            continue
        if any(re.search(p, u) for p in ignored_patterns):
            continue
        if u not in out:
            out.append(u)
    return out


def looks_like_login_wall(text: str):
    t = text or ""
    markers = [
        "Don’t miss what’s happening",
        "People on X are the first to know",
        "New to X?",
        "Sign up now to get your own personalized timeline",
    ]
    hit = sum(1 for m in markers if m in t)
    # 命中 2 个及以上且正文很短，基本可判定登录墙
    return hit >= 2 and len(t) < 1500


def build_candidate_urls(url: str):
    candidates = [url]
    m = re.search(r"/(?:i/)?status/(\d+)", url)
    if m:
        tid = m.group(1)
        candidates.extend(
            [
                f"https://x.com/i/status/{tid}",
                f"https://x.com/i/web/status/{tid}",
                f"https://twitter.com/i/status/{tid}",
            ]
        )

    # 去重保序
    out = []
    for u in candidates:
        if u not in out:
            out.append(u)
    return out


def fetch_with_browser(url: str):
    """仅使用浏览器抓取（不走任何 Twitter 第三方 API）"""
    session = f"tw-{uuid.uuid4().hex[:8]}"

    js = r"""
(() => {
  const text = document.body ? document.body.innerText : '';
  const title = document.title || '';
  const href = location.href;

  const links = Array.from(document.querySelectorAll('a[href]'))
    .map(a => a.href)
    .filter(Boolean);

  const media = [
    ...Array.from(document.querySelectorAll('img[src]')).map(x => x.src),
    ...Array.from(document.querySelectorAll('video[src]')).map(x => x.src),
    ...Array.from(document.querySelectorAll('video source[src]')).map(x => x.src),
  ].filter(Boolean);

  return {
    url: href,
    title,
    text,
    links: Array.from(new Set(links)).slice(0, 200),
    mediaURLs: Array.from(new Set(media)).slice(0, 50),
  };
})();
"""

    last_error = None

    try:
        for candidate in build_candidate_urls(url):
            try:
                run_cmd(["agent-browser", "--session", session, "open", candidate], timeout=90)
                run_cmd(
                    ["agent-browser", "--session", session, "wait", "--load", "domcontentloaded"],
                    timeout=90,
                )

                raw = run_cmd(
                    ["agent-browser", "--session", session, "eval", "--stdin"],
                    input_text=js,
                    timeout=120,
                )
                data = parse_eval_output(raw)

                # 尝试抓更多可点击内容（含 ref）作为兜底
                try:
                    snap = run_cmd(["agent-browser", "--session", session, "snapshot", "-i"], timeout=60)
                    data["snapshot"] = snap
                except Exception:
                    pass

                data["text"] = clean_text(data.get("text", ""))
                data["mediaURLs"] = clean_media_urls(data.get("mediaURLs", []))

                if data.get("text") and not looks_like_login_wall(data["text"]):
                    return data

                last_error = "命中登录墙或正文过短"
            except Exception as e:
                last_error = str(e)
                continue

        return {"error": f"页面已加载，但未提取到稳定正文（可能被登录墙或反爬策略拦截）。最后错误：{last_error or 'unknown'}"}
    except Exception as e:
        return {"error": f"浏览器抓取失败：{e}"}
    finally:
        try:
            run_cmd(["agent-browser", "--session", session, "close"], timeout=20)
        except Exception:
            pass


def infer_author(text: str, title: str):
    author_name = "Unknown"
    author_handle = "unknown"

    # 从标题猜 author: "Name on X: ..."
    m = re.search(r"^(.*?)\s+on\s+X", title or "")
    if m:
        author_name = m.group(1).strip()

    # 从正文猜 handle
    mh = re.search(r"@([A-Za-z0-9_]{2,30})", text or "")
    if mh:
        author_handle = mh.group(1)

    return author_name, author_handle


def build_three_paragraph_summary(text: str):
    """默认生成 3 段精简摘要（主旨 / 中段要点 / 结论建议）"""
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    drop_exact = {
        "Log in",
        "Sign up",
        "Article",
        "See new posts",
        "Conversation",
    }

    def is_noise(ln: str):
        if ln in drop_exact:
            return True
        if ln.startswith("@"):
            return True
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[KMB]?", ln):
            return True
        # 标题样式：一、二、三、四、➡️...
        if re.match(r"^(?:[一二三四五六七八九十]+、|➡️)", ln):
            return True
        return False

    candidates = [ln for ln in lines if len(ln) >= 14 and not is_noise(ln)]
    if not candidates:
        return []

    def pick_by_keywords(pool, keywords):
        for ln in pool:
            if any(k in ln for k in keywords):
                return ln
        return None

    # 1) 主旨：优先找“核心优势/热度/机制”
    first = pick_by_keywords(
        candidates,
        ["核心优势", "热度", "机制", "主打", "优先安装", "筛选了"],
    )
    if not first:
        first = candidates[0]

    # 2) 中段：优先找“清单/模块/能力”
    middle_pool = candidates[max(0, len(candidates) // 4): max(1, len(candidates) * 3 // 4)]
    second = pick_by_keywords(
        middle_pool,
        ["skill", "模块", "支持", "工作流", "记忆", "生产力", "投研", "github", "notion"],
    )
    if not second and middle_pool:
        second = middle_pool[len(middle_pool) // 2]

    # 3) 结论：优先找“注意事项/建议/重启/并发”
    tail_pool = candidates[len(candidates) // 2:]
    third = pick_by_keywords(
        tail_pool,
        ["关键注意事项", "建议", "必须重启", "控制并发", "才能生效", "避免"],
    )

    # 命中的是小标题时，向后找一条更“可读”的说明句
    if third and len(third) <= 16 and third.endswith("："):
        try:
            idx = candidates.index(third)
            for nxt in candidates[idx + 1:]:
                if len(nxt) >= 18 and not nxt.endswith("："):
                    third = nxt
                    break
        except Exception:
            pass

    if not third:
        third = tail_pool[-1] if tail_pool else candidates[-1]

    paras = []
    for p in [first, second, third]:
        if p and p not in paras:
            paras.append(p)

    # 不足三条时补齐
    for ln in candidates:
        if len(paras) >= 3:
            break
        if ln not in paras:
            paras.append(ln)

    return [p[:220] + ("…" if len(p) > 220 else "") for p in paras[:3]]


def format_deepreeder_md(data, original_url):
    """生成结构化 Markdown 输出"""
    if "error" in data:
        return f"❌ [抓取失败]({original_url}): {data['error']}\n"

    text = data.get("text", "")
    title = data.get("title", "")
    author_name, author_handle = infer_author(text, title)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    summary = build_three_paragraph_summary(text)

    md = f"""---
author: "@{author_handle}"
source: "{original_url}"
date: "{date_str}"
content_hash: "{content_hash}"
---

### 🐦 {author_name} (@{author_handle})
🕒 时间: {date_str}
📌 页面标题: {title}
"""

    if summary:
        md += "\n**🧾 三段精简摘要：**\n"
        for i, p in enumerate(summary, 1):
            md += f"{i}. {p}\n"

    md += f"\n> {text.replace(chr(10), chr(10) + '> ')}\n"

    media_list = data.get("mediaURLs", [])
    if media_list:
        md += "\n**📸 附带媒体 (Media):**\n"
        for i, url in enumerate(media_list):
            if any(x in url for x in [".mp4", "video"]):
                md += f"- 🎥 [点击查看视频源文件]({url})\n"
            else:
                md += f"![Image_{i}]({url})\n"

    return md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate Twitter Fetcher (Browser-first)")
    parser.add_argument("--urls", required=True, help="包含 X/Twitter 链接的一段文本")
    args = parser.parse_args()

    urls = extract_urls(args.urls)
    if not urls:
        print("⚠️ 未在输入中检测到有效的 Twitter/X 链接。")
        exit(1)

    print("### 🔍 抓取结果（Browser-first）\n")
    for url in urls:
        data = fetch_with_browser(url)
        print(format_deepreeder_md(data, url))
        print("\n---\n")
