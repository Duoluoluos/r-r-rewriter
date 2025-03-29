import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import re
import tiktoken
import time
import os 


def extract_text_from_html(html, max_tokens=10000, encoding_name="cl100k_base"):
    """
    从 HTML 内容中提取核心文本信息，去除无关内容；
    如果最终 token 数目大于 max_tokens，则进行截断。
    
    参数:
      html: 网页的 HTML 源码（字符串）。
      max_tokens: 最大允许的 token 数目（默认10000）。
      encoding_name: tiktoken 使用的编码名称，默认使用 "cl100k_base"（适用于 GPT-4、GPT-3.5 等）。
      
    返回:
      一个字符串，包含提取后的核心文本（经过截断，如果 tokens 数超过上限）。
    """
    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # 移除不需要的标签，如 script、style、header、footer、nav、aside
    for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
        tag.decompose()
    
    # 尝试聚焦在文章的主要部分：优先提取 <article>、<main> 或带有 content/body 关键字的容器
    main_content = soup.find(
        ['article', 'main', 'div', 'section'], 
        class_=re.compile('.*(content|body).*', re.IGNORECASE)
    )
    if main_content:
        soup = main_content

    # 提取纯文本，使用换行符分割并移除多余空白
    text = soup.get_text(separator='\n', strip=True)
    text = re.sub(r'(\n|\s)+', '\n', text)  # 规范换行和空格
    text = re.sub(r'Copyright.*|Privacy.*|Terms of Service.*|Cookie Policy.*', '', text, flags=re.IGNORECASE)
    
    # 过滤空行
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = ' '.join(lines)

    # 使用 tiktoken 进行 token 化和截断
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    
    return text

def search_web(query, k=10):
    """
    使用 DuckDuckGo 搜索并获取前 k 个搜索结果的网页内容，然后提取有效文字信息。
    
    参数:
      query: 搜索关键词（字符串）。
      k: 返回的网页数量（默认为10）。
      
    返回:
      一个列表，每个元素为字典，包含 'url'、'raw_content'（原始HTML）和 'text'（提取的纯文本）。
    """
    results = []
    # 使用新版 DuckDuckGo 搜索接口（基于 DDGS 类）
    with DDGS() as ddgs:
        # ddgs.text 返回的是生成器，每个结果包含 'title', 'href', 'body' 等信息
        for r in ddgs.text(query, max_results=k):
            results.append(r)
    
    if not results:
        print("未搜索到相关结果。")
        return []
    
    # 提取搜索结果中的链接地址（可能有的结果没有 'href' 字段）
    top_urls = [item.get('href') for item in results if item.get('href')][:k]
    
    fetched_results = []
    for url in top_urls:
        try:
            #print(f"正在获取：{url}")
            # 设置请求头，模拟浏览器访问
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
                )
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                raw_html = response.text
                text = extract_text_from_html(raw_html)
            else:
                raw_html = ""
                text = f"无法获取内容，状态码：{response.status_code}"
        except Exception as e:
            raw_html = ""
            text = f"请求错误：{e}"
        
        fetched_results.append({
            'url': url,
            'raw_content': raw_html,
            'text': text
        })
        
        # 添加延迟，避免频繁请求
        time.sleep(1) 
    
    return fetched_results


