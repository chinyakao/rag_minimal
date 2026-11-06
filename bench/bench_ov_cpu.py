"""找出 OpenVINO CPU 為什麼比 PyTorch CPU 慢: 一次只改一個變因的對照實驗.

第一輪結果 (bench_embed.py): 建索引 PyTorch 48.8 chunks/s vs OpenVINO CPU 40.7;
單筆查詢 PyTorch 12.2 ms vs OpenVINO CPU 24.7 ms. 兩個假設:

1. 補齊 (padding): OpenVINO 為了 NPU 固定成 [batch, 128], 短的查詢也補到 128 token;
   PyTorch 只補到這批最長. 查詢通常只有 10 到 30 個 token, 等於多算好幾倍.
2. 用了幾顆核心: Meteor Lake 有 P-core, E-core, LP E-core. OpenVINO CPU 預設的延遲模式
   在混合架構上可能只用 P-core; PyTorch 用所有核心.

變體 (都是 CPU, 都不需要 sudo):
    st          PyTorch (sentence-transformers) 基準
    ov-static   OpenVINO 固定 [8,128] / [1,128], 預設設定  (第一輪的做法)
    ov-dynamic  OpenVINO 不固定形狀, 只補到該批最長          (驗證假設 1)
    ov-static-any   固定形狀 + SCHEDULING_CORE_TYPE=ANY_CORE  (驗證假設 2)
    ov-dynamic-any  兩個都改

用法 (在 repo 根目錄, 先跑過 rag.py index):
    DATA_DIR=data/oem-7.0 python bench/bench_ov_cpu.py
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import rag  # noqa: E402
from bench_embed import load_corpus, load_queries  # noqa: E402
from ownership import give_back  # noqa: E402

# 第一輪結果: ANY_CORE 本來就是預設, 四個 OpenVINO 變體都只用 6 個執行緒 (PyTorch 用 16).
# 第二輪改測: 執行緒數, 以及「查詢用短一點的固定長度」(NPU 也能用的解法)
VARIANTS = {
    "ov-static": dict(dynamic=False, config={}),
    "ov-static-q32": dict(dynamic=False, config={}, query_seq_len=32),
    "ov-dynamic": dict(dynamic=True, config={}),
    "ov-static-t16": dict(dynamic=False, config={"INFERENCE_NUM_THREADS": 16}),
    "ov-dynamic-t16": dict(dynamic=True, config={"INFERENCE_NUM_THREADS": 16}),
}


def token_lengths(texts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rag.EMBED_MODEL)
    lens = [len(tok(t, truncation=True, max_length=128)["input_ids"]) for t in texts]
    return float(np.mean(lens)), float(np.percentile(lens, 95))


def run_loop(fn, min_seconds):
    t0, n = time.perf_counter(), 0
    while True:
        fn()
        n += 1
        if time.perf_counter() - t0 >= min_seconds:
            return n, time.perf_counter() - t0


def make(name, batch, query=False):
    if name == "st":
        import torch
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(rag.EMBED_MODEL, device="cpu")
        enc = lambda texts: np.asarray(m.encode(texts, batch_size=batch, normalize_embeddings=True))  # noqa: E731
        return enc, {"torch_threads": torch.get_num_threads()}
    from embed_ov import OVEmbedder
    v = dict(VARIANTS[name])
    q_len = v.pop("query_seq_len", 128)
    m = OVEmbedder(rag.EMBED_MODEL, device="CPU", batch=batch, use_cache=False,
                   seq_len=q_len if query else 128, **v)
    return m.encode, m.properties()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["st", *VARIANTS])
    ap.add_argument("--n-chunks", type=int, default=256)
    ap.add_argument("--min-seconds", type=float, default=8.0)
    ap.add_argument("--query-iters", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=3, help="建索引測試重複幾次, 取中位數")
    args = ap.parse_args()

    corpus, queries = load_corpus(args.n_chunks), load_queries()
    c_mean, c_p95 = token_lengths(corpus)
    q_mean, q_p95 = token_lengths(queries)
    print(f"cpu_count={os.cpu_count()}  chunk token 平均 {c_mean:.0f} (p95 {c_p95:.0f})  "
          f"查詢 token 平均 {q_mean:.0f} (p95 {q_p95:.0f})  固定形狀一律 128")

    rows, ref = [], None
    for name in args.variants:
        print(f"\n== {name} ==")
        enc, props = make(name, 8)
        enc(corpus[:16])  # 暖機
        cps = []
        for _ in range(args.repeat):  # 同一設定重複幾次, 看誤差有多大
            n, secs = run_loop(lambda: enc(corpus), args.min_seconds)
            cps.append(n * len(corpus) / secs)
        vecs = enc(corpus[:64])
        ref = vecs if ref is None else ref
        q_enc, _ = make(name, 1, query=True)
        for q in queries[:5]:
            q_enc([q])
        lat = []
        for i in range(args.query_iters):
            t0 = time.perf_counter()
            q_enc([queries[i % len(queries)]])
            lat.append((time.perf_counter() - t0) * 1000)
        row = {"variant": name, "index_cps": float(np.median(cps)),
               "index_cps_min": min(cps), "index_cps_max": max(cps),
               "query_p50_ms": float(np.percentile(lat, 50)), "query_p95_ms": float(np.percentile(lat, 95)),
               "cos_vs_first": float(np.mean(np.sum(vecs * ref, axis=1))), **props}
        rows.append(row)
        print(row)

    print(f"\n{'variant':>15} | {'index cps (min-max)':>20} | {'query p50':>9} | {'p95':>7} | threads")
    for r in rows:
        extra = r.get("torch_threads") or r.get("INFERENCE_NUM_THREADS", "?")
        rng = f"{r['index_cps']:.1f} ({r['index_cps_min']:.0f}-{r['index_cps_max']:.0f})"
        print(f"{r['variant']:>15} | {rng:>20} | {r['query_p50_ms']:7.1f}ms | {r['query_p95_ms']:5.1f}ms | {extra}")

    meta = {"time": datetime.now().isoformat(timespec="seconds"), "cpu_count": os.cpu_count(),
            "chunk_tokens_mean": c_mean, "query_tokens_mean": q_mean}
    out = ROOT / "bench" / "results" / f"ov_cpu_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, **meta}, ensure_ascii=False) + "\n")
    give_back(out.parent)
    print(f"\n結果已存到 {out}")


if __name__ == "__main__":
    main()
