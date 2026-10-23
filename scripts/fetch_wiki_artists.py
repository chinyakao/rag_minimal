#!/usr/bin/env python3
"""從中文維基百科下載 20 位台灣藝人條目, 轉成繁體中文 Markdown.

安裝依賴:  pip install requests
執行:      WIKI_CONTACT=you@example.com python scripts/fetch_wiki_artists.py
輸出:      <repo>/data/<條目名>.md  (data/ 內容不進版控, 見 .gitignore)

維基百科文字採 CC BY-SA 4.0 授權, 每個檔案開頭都附原網址與授權說明.
"""
import os
import re
import time
from urllib.parse import quote

import requests

TITLES = [
    "周杰倫", "蔡依林", "張惠妹", "田馥甄", "陳綺貞",
    "伍佰", "蕭敬騰", "羅志祥", "楊丞琳", "林志玲",
    "王心凌", "盧廣仲", "羅大佑", "鄧麗君", "李宗盛",
    "劉若英", "張韶涵", "林依晨", "金城武", "周渝民",
]

API = "https://zh.wikipedia.org/w/api.php"
# Wikimedia 要求帶有可辨識的 User-Agent: 請用環境變數 WIKI_CONTACT 提供你的聯絡方式 (不要寫進程式碼)
CONTACT = os.environ.get("WIKI_CONTACT", "your_email@example.com")
HEADERS = {"User-Agent": f"rag-practice-script/1.0 ({CONTACT})"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def fetch(title: str) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|revisions",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "redirects": 1,
        "variant": "zh-tw",  # 轉成臺灣正體用字
        "rvprop": "ids",
        "titles": title,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    return next(iter(pages.values()))


def wiki_to_md(text: str) -> str:
    # "== 標題 ==" → "## 標題", 層級依等號數量
    def repl(m):
        level = len(m.group(1))
        return "#" * max(level, 2) + " " + m.group(2).strip()

    text = re.sub(r"^(={2,6})\s*(.+?)\s*\1\s*$", repl, text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    for t in TITLES:
        try:
            page = fetch(t)
            if "missing" in page or not page.get("extract"):
                print(f"! 找不到條目:{t}")
                continue
            title = page["title"]
            revid = page["revisions"][0]["revid"]
            url = f"https://zh.wikipedia.org/zh-tw/{quote(title)}"
            permalink = f"https://zh.wikipedia.org/w/index.php?oldid={revid}"
            body = wiki_to_md(page["extract"])
            md = (
                f"# {title}\n\n"
                f"> 原始網址:<{url}>\n"
                f"> 版本永久連結:<{permalink}>\n"
                f"> 授權: CC BY-SA 4.0 (維基百科貢獻者)\n\n"
                f"---\n\n{body}\n"
            )
            with open(os.path.join(OUT, f"{title}.md"), "w", encoding="utf-8") as f:
                f.write(md)
            print(f"✓ {title} ({len(body)} 字)")
        except Exception as e:
            print(f"! 失敗 {t}: {e}")
        time.sleep(1)
    print(f"完成, 輸出在 {os.path.normpath(OUT)}/")


if __name__ == "__main__":
    main()
