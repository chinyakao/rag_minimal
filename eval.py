"""檢索評估: Ubuntu linux-oem-7.0 kernel changelog, 28 題 (題庫在 eval_oem.py)

用法:
    python eval.py            # 只評估檢索 (不呼叫 LLM, 不花 API 費用), k=3
    python eval.py 5          # 改用 k=5
    python eval.py --llm      # 另外呼叫 LLM, 檢查 refuse 類題目是否拒答
    python eval.py --bm25     # 檢索改用 BM25; --hybrid 則用向量 + BM25 的 RRF 融合 (預設只用向量)
    BM25_WEIGHT=2 python eval.py --hybrid   # 加權 RRF, BM25 權重 2, 向量 1
    (--oem 仍可加, 但已經是預設, 不影響結果)

欄位說明:
    type             題型 (single / multi_para / cross / reasoning / boundary / paraphrase)
    expected_sources 應出現在檢索結果中的文件 (可多個). 空 list 表示不做 hit 判斷
    expect           answer = 資料中有答案; refuse = 資料中找不到, 應拒答;
                     manual = 需人工看回答 (例如推理題)
    note             備註, 只在輸出中顯示

早期的中文藝人題庫與 Kubernetes 題庫已封存在 archive/, 見 archive/README.md.
"""
import os
import sys

# --bm25 / --hybrid: 切換檢索方式, 要在 import rag 之前設定
for _mode in ("bm25", "hybrid"):
    if f"--{_mode}" in sys.argv:
        os.environ["RETRIEVAL"] = _mode

from eval_oem import EVAL_SET_OEM as EVAL_SET
from rag import BM25_WEIGHT, RETRIEVAL, retrieve

REFUSAL_TEXT = "資料中找不到答案"


def run(k=3, use_llm=False):
    hit_total = hit_ok = 0
    by_type = {}
    refuse_total = refuse_ok = 0

    for item in EVAL_SET:
        results = retrieve(item["q"], k=k)
        sources = [r["source"] for _, r in results]
        top1 = results[0][0]
        exp = item["expected_sources"]
        tag = f"#{item['id']:<2} [{item['type']}]"

        if exp:
            found = [s for s in exp if s in sources]
            ok = len(found) > 0
            recall = len(found) / len(exp)
            hit_total += 1
            hit_ok += ok
            t = by_type.setdefault(item["type"], [0, 0])
            t[0] += 1
            t[1] += ok
            status = "OK  " if ok else "MISS"
            extra = f" recall={len(found)}/{len(exp)}" if len(exp) > 1 else ""
            print(f"{status} {tag}{extra} {item['q']}")
            print(f"       -> {sources} (top1={top1:.3f})")
        else:
            print(f"---- {tag} {item['q']}")
            print(f"       -> {sources} (top1={top1:.3f})")

        if item.get("note"):
            print(f"       note: {item['note']}")

        if use_llm and item["expect"] in ("refuse", "manual"):
            from rag import build_prompt, generate
            answer, _, _ = generate(build_prompt(item["q"], results))
            if item["expect"] == "refuse":
                refused = REFUSAL_TEXT in answer
                refuse_total += 1
                refuse_ok += refused
                print(f"       LLM: {'REFUSED' if refused else 'ANSWERED (可能幻覺)'}")
            else:
                print(f"       LLM 回答 (請人工檢查): {answer[:200]}...")

    mode = f"hybrid, BM25_WEIGHT={BM25_WEIGHT:g}" if RETRIEVAL == "hybrid" else RETRIEVAL
    print(f"\n[{mode}] hit@{k} = {hit_ok}/{hit_total} = {hit_ok / hit_total:.2f}")
    for name, (n, ok) in by_type.items():
        print(f"  {name:<10} {ok}/{n}")
    if use_llm and refuse_total:
        print(f"拒答率 = {refuse_ok}/{refuse_total}")


if __name__ == "__main__":
    args = sys.argv[1:]
    k = next((int(a) for a in args if a.isdigit()), 3)
    run(k=k, use_llm="--llm" in args)
