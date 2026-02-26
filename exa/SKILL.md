---
name: exa
description: Exa AI 语义搜索引擎。当用户需要深度搜索、找代码示例、研究公司、查找论文、搜索推文时使用。比普通搜索引擎更精准，支持语义理解和内容提取。
version: 1.0.0
---

# Exa AI 搜索引擎

## 🎯 触发场景

- 用户说"搜索 XXX"、"找一下 XXX"
- 需要代码示例、技术文档
- 研究公司、行业信息
- 查找学术论文、研究资料
- 搜索推文/X讨论
- 需要比普通搜索更精准的结果

## ⚙️ 使用方法

```bash
# 基础搜索
python3 exa/search.py "OpenClaw AI agent"

# 限制结果数量
python3 exa/search.py "Claude Code MCP" --num 5

# 按类别搜索
python3 exa/search.py "Anthropic" --category company
python3 exa/search.py "GPT-5" --category news
python3 exa/search.py "transformer attention" --category "research paper"

# 实时抓取（获取最新内容）
python3 exa/search.py "最新的 AI 新闻" --livecrawl
```

## 📋 支持的类别

| 类别 | 用途 |
|------|------|
| `company` | 公司信息、融资、竞品 |
| `news` | 新闻报道、公告 |
| `research paper` | 学术论文、arXiv |
| `tweet` | 推文、X 讨论 |
| `personal site` | 个人博客、技术文章 |
| `financial report` | 财报、SEC 文件 |

## 🔧 自动使用场景

当用户请求包含以下关键词时，自动调用 Exa 搜索：
- "搜索"、"查找"、"找一下"
- "代码示例"、"how to"
- "公司"、"融资"、"竞品分析"
- "论文"、"研究"、"文献"
- "最新"、"新闻"、"动态"

## 💡 优势

- **语义理解**：不只是关键词匹配，理解搜索意图
- **内容提取**：自动提取页面正文，不用点链接
- **高亮显示**：显示最相关的段落
- **多类别**：针对不同场景优化搜索范围

## ⚠️ 注意

- API 有调用限制，合理使用
- 复杂查询建议拆分成多次搜索
- 结果自动格式化为 Markdown，适合直接展示
