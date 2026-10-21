"""最小 RAG: 不用 LangChain, 只用 numpy + sentence-transformers + Anthropic SDK.

用法:
    python rag.py index                      # 讀 docs/ 建索引
    python rag.py ask "你的問題"              # 檢索 + 生成
    python rag.py ask "你的問題" --no-llm     # 只看檢索結果 (不需 API key)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# 從同資料夾的 .env 讀取 ANTHROPIC_API_KEY 等設定; 沒有 .env 或沒裝套件時略過
try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass

DOCS_DIR = HERE / "docs"
INDEX_FILE = HERE / "index.npz"
CHUNKS_FILE = HERE / "chunks.json"

# 多語言模型, 中英文皆可; 可用環境變數替換
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")

CHUNK_SIZE = 400      # 字元數; 之後可改成 token 數並做實驗
CHUNK_OVERLAP = 80
TOP_K = 4

_model = None


def get_model():
    """延遲載入 embedding 模型 (載入很慢, 只做一次)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ---------- Step 1: 載入與切塊 ----------
def load_docs():
    docs = []
    for p in sorted(DOCS_DIR.glob("*")):
        if p.suffix.lower() in {".md", ".txt"}:
            docs.append((p.name, p.read_text(encoding="utf-8")))
    return docs


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """固定長度滑動視窗. 簡單但會切斷語意, 這是你之後要改進的點."""
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
        start += size - overlap
    return chunks


# ---------- Step 2: 嵌入與建索引 ----------
def embed(texts):
    vecs = get_model().encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def build_index():
    records = []
    for name, text in load_docs():
        for i, c in enumerate(chunk_text(text)):
            records.append({"source": name, "chunk_id": i, "text": c})
    if not records:
        sys.exit("docs/ 內沒有 .md 或 .txt 檔")
    t0 = time.time()
    matrix = embed([r["text"] for r in records])
    np.savez(INDEX_FILE, matrix=matrix)
    CHUNKS_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已索引 {len(records)} 個 chunk, 耗時 {time.time() - t0:.2f}s")


# ---------- Step 3: 檢索 ----------
def load_index():
    if not INDEX_FILE.exists():
        sys.exit("找不到索引, 請先執行: python rag.py index")
    matrix = np.load(INDEX_FILE)["matrix"]
    records = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return matrix, records


def retrieve(query, k=TOP_K):
    matrix, records = load_index()
    q = embed([query])[0]
    scores = matrix @ q  # 已正規化, 內積即 cosine similarity
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), records[i]) for i in top]


# ---------- Step 4: 組 prompt 與生成 ----------
def build_prompt(question, hits):
    ctx = "\n\n".join(
        f"[{n}] (來源: {r['source']})\n{r['text']}" for n, (_, r) in enumerate(hits, 1)
    )
    return (
        "你是一個只根據提供資料回答的助理.\n"
        "規則:\n"
        "1. 只使用下方資料回答, 並在句尾用 [編號] 標註引用.\n"
        "2. 若資料不足以回答, 直接說「資料中找不到答案」, 不要猜測.\n\n"
        f"資料:\n{ctx}\n\n問題: {question}"
    )


def generate(prompt):
    import anthropic
    client = anthropic.Anthropic()  # 讀取 ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = msg.usage
    return msg.content[0].text, usage.input_tokens, usage.output_tokens


# ---------- Step 5: 串起來並量測延遲 ----------
def ask(question, use_llm=True):
    t0 = time.time()
    hits = retrieve(question)
    t_retrieve = time.time() - t0

    print("== 檢索結果 ==")
    for n, (score, r) in enumerate(hits, 1):
        preview = r["text"][:80].replace("\n", " ")
        print(f"[{n}] score={score:.3f} {r['source']}#{r['chunk_id']}: {preview}...")

    if not use_llm:
        print(f"\n檢索耗時 {t_retrieve * 1000:.0f}ms (含查詢嵌入)")
        return

    prompt = build_prompt(question, hits)
    t1 = time.time()
    answer, tin, tout = generate(prompt)
    t_llm = time.time() - t1

    print("\n== 回答 ==")
    print(answer)
    print(
        f"\n檢索 {t_retrieve * 1000:.0f}ms | LLM {t_llm * 1000:.0f}ms | "
        f"input tokens={tin} output tokens={tout}"
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    if args.cmd == "index":
        build_index()
    else:
        ask(args.question, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
