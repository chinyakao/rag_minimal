"""最小 RAG: 不用 LangChain, 只用 numpy + sentence-transformers + Anthropic SDK.

用法:
    python rag.py index                      # 讀 data/oem-7.0/ 建索引
    python rag.py ask "你的問題"              # 檢索 + 生成
    python rag.py ask "你的問題" --no-llm     # 只看檢索結果 (不需 API key)
    RETRIEVAL=hybrid python rag.py ask "..."  # 檢索方式: vector (預設) / bm25 / hybrid
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# 從同資料夾的 .env 讀取 ANTHROPIC_API_KEY 等設定; 沒有 .env 或沒裝套件時略過
try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass

# 預設資料集: Ubuntu linux-oem-7.0 kernel changelog; 可用環境變數 DATA_DIR 切換
DATA_DIR = Path(os.environ.get("DATA_DIR", HERE / "data" / "oem-7.0"))
# 每個資料集各自一份索引 (檔名帶資料夾名, 例如 index_oem-7.0.npz), 切換 DATA_DIR 不會互相覆蓋
_SUFFIX = f"_{DATA_DIR.name}"
# CHUNK_TITLE=1: 每個 chunk 前面加上文件標題 (第一行的 "# 標題", 沒有就用檔名).
# 例如 changelog 的版本號只寫在檔案開頭, 加上後每個 chunk 都知道自己屬於哪個版本.
# 開啟時索引另存 (檔名加 _title), 不會蓋掉沒加標題的索引, 方便比較.
CHUNK_TITLE = os.environ.get("CHUNK_TITLE", "0") == "1"
if CHUNK_TITLE:
    _SUFFIX += "_title"
INDEX_FILE = HERE / f"index{_SUFFIX}.npz"
CHUNKS_FILE = HERE / f"chunks{_SUFFIX}.json"

# 多語言模型, 中英文皆可; 可用環境變數替換
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")

# embedding 後端: st = sentence-transformers (預設, PyTorch);
# openvino = embed_ov.py, 可用 EMBED_DEVICE 指定 CPU / GPU / NPU
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "st")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "CPU")

CHUNK_SIZE = 400      # 字元數; 之後可改成 token 數並做實驗
CHUNK_OVERLAP = 80
TOP_K = 4

# 檢索方式: vector = 只用向量 (預設); bm25 = 只用關鍵字; hybrid = 兩者用 RRF 融合
RETRIEVAL = os.environ.get("RETRIEVAL", "vector")
BM25_K1, BM25_B = 1.5, 0.75   # BM25 常用預設值
RRF_K = 60                    # RRF 常用預設值, 越大則排名前後的差距越小
# 加權 RRF: BM25 那一邊的分數乘上這個權重 (向量固定為 1). 預設 1 = 原本的等權重.
# 例如 BM25_WEIGHT=2: 某個 chunk 在 BM25 排第 1 的貢獻是 2/61, 比向量排第 1 的 1/61 大一倍.
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "1"))
CANDIDATES = 50               # hybrid 時兩邊各取前幾名來融合

_model = None


def get_model():
    """延遲載入 embedding 模型 (載入很慢, 只做一次)."""
    global _model
    if _model is None:
        if EMBED_BACKEND == "openvino":
            from embed_ov import OVEmbedder
            _model = OVEmbedder(EMBED_MODEL, device=EMBED_DEVICE)
        else:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ---------- Step 1: 載入與切塊 ----------
def load_docs():
    docs = []
    for p in sorted(DATA_DIR.glob("*")):
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
    model = get_model()
    if EMBED_BACKEND == "openvino":
        vecs = model.encode(texts)  # 已在 embed_ov.py 內正規化
    else:
        vecs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def doc_title(name, text):
    """文件標題: 第一行是 Markdown "# 標題" 就用它, 否則用檔名 (去掉副檔名)."""
    first = text.lstrip().split("\n", 1)[0]
    return first[2:].strip() if first.startswith("# ") else Path(name).stem


def build_index():
    records = []
    for name, text in load_docs():
        title = doc_title(name, text)
        for i, c in enumerate(chunk_text(text)):
            if CHUNK_TITLE:
                c = f"[{title}]\n{c}"
            records.append({"source": name, "chunk_id": i, "text": c})
    if not records:
        sys.exit(f"{DATA_DIR} 內沒有 .md 或 .txt 檔")
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
    return matrix, load_records()


def load_records():
    if not CHUNKS_FILE.exists():
        sys.exit("找不到索引, 請先執行: python rag.py index")
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def vector_scores(query, matrix):
    q = embed([query])[0]
    return matrix @ q  # 已正規化, 內積即 cosine similarity


# ---------- Step 3b: BM25 關鍵字檢索 ----------
# 英文/數字: 小寫, 保留 "0xc05", "ov02c10", "7.0.0", "1018.18" 這種中間有 . 或 _ 的字;
# 中文沒有空格, 用相鄰兩字 (bigram) 當詞, 不需要斷詞套件.
_WORD = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text):
    text = text.lower()
    tokens = _WORD.findall(text)
    for run in _CJK.findall(text):
        tokens += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    return tokens


class BM25:
    """手寫 BM25. 分數 = sum over 查詢詞 of idf * tf*(k1+1) / (tf + k1*(1-b+b*長度/平均長度))."""

    def __init__(self, texts, k1=BM25_K1, b=BM25_B):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=np.float32)
        self.avg_len = float(self.lens.mean())
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(texts)
        # 出現在越少 chunk 的詞越有鑑別力 (例如 "0xc05"), "the" 這種詞幾乎是 0
        self.idf = {t: float(np.log(1 + (n - d + 0.5) / (d + 0.5))) for t, d in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.tfs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lens / self.avg_len)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.tfs], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


_bm25 = None


def get_bm25(records):
    """BM25 不需另存索引, 直接從 chunks 建 (幾千個 chunk 不到 1 秒), 只建一次."""
    global _bm25
    if _bm25 is None:
        _bm25 = BM25([r["text"] for r in records])
    return _bm25


def rrf(rankings, weights=None, k=RRF_K):
    """Reciprocal Rank Fusion: 每個 chunk 的分數 = sum 權重/(k + 名次). 只看名次, 不看原始分數,
    所以 cosine (0~1) 和 BM25 (0~幾十) 不用先調成同一個尺度."""
    weights = weights or [1.0] * len(rankings)
    fused = {}
    for ranking, w in zip(rankings, weights):
        for rank, i in enumerate(ranking, 1):
            fused[i] = fused.get(i, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def retrieve(query, k=TOP_K, mode=None):
    """回傳 [(分數, record)]. 分數的意義依 mode 而不同: cosine / BM25 / RRF."""
    mode = mode or RETRIEVAL
    if mode == "bm25":
        records = load_records()
        scores = get_bm25(records).scores(query)
    elif mode == "vector":
        matrix, records = load_index()
        scores = vector_scores(query, matrix)
    elif mode == "hybrid":
        matrix, records = load_index()
        n = min(CANDIDATES, len(records))
        by_vec = np.argsort(-vector_scores(query, matrix))[:n]
        by_bm25 = np.argsort(-get_bm25(records).scores(query))[:n]
        fused = rrf([by_vec, by_bm25], weights=[1.0, BM25_WEIGHT])
        return [(s, records[i]) for i, s in fused[:k]]
    else:
        sys.exit(f"未知的 RETRIEVAL={mode}, 可用 vector / bm25 / hybrid")
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
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = msg.usage
    # 回應的 content 是多個區塊 (可能先有 thinking 區塊, 再有 text 區塊), 只取 text
    text = "".join(b.text for b in msg.content if b.type == "text")
    if not text:
        sys.exit(f"模型沒有回傳文字 (stop_reason={msg.stop_reason}), 可嘗試調高 max_tokens")
    return text, usage.input_tokens, usage.output_tokens


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
        print(f"\n檢索耗時 {t_retrieve * 1000:.0f}ms ({RETRIEVAL}, 第一次查詢含模型與 BM25 載入)")
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
