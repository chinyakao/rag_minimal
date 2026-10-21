"""簡易檢索評估: 計算 hit@k (正確來源文件是否出現在前 k 個結果)

在 EVAL_SET 補上你自己的題目; expected_source 填文件檔名
先從 20 題開始, 之後擴充到 50 題並加入刁難題
"""
from rag import retrieve

EVAL_SET = [
    {"q": "cgroups 可以限制什麼資源?", "expected_source": "linux_isolation.md"},
    {"q": "namespace 的用途是什麼?", "expected_source": "linux_isolation.md"},
    {"q": "RAG 為什麼要切塊?", "expected_source": "rag_basics.md"},
    {"q": "什麼是 hybrid search?", "expected_source": "rag_basics.md"},
]


def run(k=3):
    hits_ok = 0
    for item in EVAL_SET:
        results = retrieve(item["q"], k=k)
        sources = [r["source"] for _, r in results]
        ok = item["expected_source"] in sources
        hits_ok += ok
        print(f"{'OK  ' if ok else 'MISS'} {item['q']} -> {sources}")
    print(f"\nhit@{k} = {hits_ok}/{len(EVAL_SET)} = {hits_ok / len(EVAL_SET):.2f}")


if __name__ == "__main__":
    run()
